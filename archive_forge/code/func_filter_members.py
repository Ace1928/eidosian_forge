import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence,
from docutils.statemachine import StringList
import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import (get_class_members, get_object_members, import_module,
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
from sphinx.util.typing import OptionSpec, get_type_hints, restify
from sphinx.util.typing import stringify as stringify_typehint
def filter_members(self, members: ObjectMembers, want_all: bool) -> List[Tuple[str, Any, bool]]:
    """Filter the given member list.

        Members are skipped if

        - they are private (except if given explicitly or the private-members
          option is set)
        - they are special methods (except if given explicitly or the
          special-members option is set)
        - they are undocumented (except if the undoc-members option is set)

        The user can override the skipping decision by connecting to the
        ``autodoc-skip-member`` event.
        """

    def is_filtered_inherited_member(name: str, obj: Any) -> bool:
        inherited_members = self.options.inherited_members or set()
        if inspect.isclass(self.object):
            for cls in self.object.__mro__:
                if cls.__name__ in inherited_members and cls != self.object:
                    return True
                elif name in cls.__dict__:
                    return False
                elif name in self.get_attr(cls, '__annotations__', {}):
                    return False
                elif isinstance(obj, ObjectMember) and obj.class_ is cls:
                    return False
        return False
    ret = []
    namespace = '.'.join(self.objpath)
    if self.analyzer:
        attr_docs = self.analyzer.find_attr_docs()
    else:
        attr_docs = {}
    for obj in members:
        try:
            membername, member = obj
            if member is INSTANCEATTR:
                isattr = True
            elif (namespace, membername) in attr_docs:
                isattr = True
            else:
                isattr = False
            doc = getdoc(member, self.get_attr, self.config.autodoc_inherit_docstrings, self.object, membername)
            if not isinstance(doc, str):
                doc = None
            cls = self.get_attr(member, '__class__', None)
            if cls:
                cls_doc = self.get_attr(cls, '__doc__', None)
                if cls_doc == doc:
                    doc = None
            if isinstance(obj, ObjectMember) and obj.docstring:
                doc = obj.docstring
            doc, metadata = separate_metadata(doc)
            has_doc = bool(doc)
            if 'private' in metadata:
                isprivate = True
            elif 'public' in metadata:
                isprivate = False
            else:
                isprivate = membername.startswith('_')
            keep = False
            if ismock(member) and (namespace, membername) not in attr_docs:
                pass
            elif self.options.exclude_members and membername in self.options.exclude_members:
                keep = False
            elif want_all and special_member_re.match(membername):
                if self.options.special_members and membername in self.options.special_members:
                    if membername == '__doc__':
                        keep = False
                    elif is_filtered_inherited_member(membername, obj):
                        keep = False
                    else:
                        keep = has_doc or self.options.undoc_members
                else:
                    keep = False
            elif (namespace, membername) in attr_docs:
                if want_all and isprivate:
                    if self.options.private_members is None:
                        keep = False
                    else:
                        keep = membername in self.options.private_members
                else:
                    keep = True
            elif want_all and isprivate:
                if has_doc or self.options.undoc_members:
                    if self.options.private_members is None:
                        keep = False
                    elif is_filtered_inherited_member(membername, obj):
                        keep = False
                    else:
                        keep = membername in self.options.private_members
                else:
                    keep = False
            elif self.options.members is ALL and is_filtered_inherited_member(membername, obj):
                keep = False
            else:
                keep = has_doc or self.options.undoc_members
            if isinstance(obj, ObjectMember) and obj.skipped:
                keep = False
            if self.env.app:
                skip_user = self.env.app.emit_firstresult('autodoc-skip-member', self.objtype, membername, member, not keep, self.options)
                if skip_user is not None:
                    keep = not skip_user
        except Exception as exc:
            logger.warning(__('autodoc: failed to determine %s.%s (%r) to be documented, the following exception was raised:\n%s'), self.name, membername, member, exc, type='autodoc')
            keep = False
        if keep:
            ret.append((membername, member, isattr))
    return ret