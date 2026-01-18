import os
import typing
import typing as t
import weakref
from collections import ChainMap
from functools import lru_cache
from functools import partial
from functools import reduce
from types import CodeType
from markupsafe import Markup
from . import nodes
from .compiler import CodeGenerator
from .compiler import generate
from .defaults import BLOCK_END_STRING
from .defaults import BLOCK_START_STRING
from .defaults import COMMENT_END_STRING
from .defaults import COMMENT_START_STRING
from .defaults import DEFAULT_FILTERS
from .defaults import DEFAULT_NAMESPACE
from .defaults import DEFAULT_POLICIES
from .defaults import DEFAULT_TESTS
from .defaults import KEEP_TRAILING_NEWLINE
from .defaults import LINE_COMMENT_PREFIX
from .defaults import LINE_STATEMENT_PREFIX
from .defaults import LSTRIP_BLOCKS
from .defaults import NEWLINE_SEQUENCE
from .defaults import TRIM_BLOCKS
from .defaults import VARIABLE_END_STRING
from .defaults import VARIABLE_START_STRING
from .exceptions import TemplateNotFound
from .exceptions import TemplateRuntimeError
from .exceptions import TemplatesNotFound
from .exceptions import TemplateSyntaxError
from .exceptions import UndefinedError
from .lexer import get_lexer
from .lexer import Lexer
from .lexer import TokenStream
from .nodes import EvalContext
from .parser import Parser
from .runtime import Context
from .runtime import new_context
from .runtime import Undefined
from .utils import _PassArg
from .utils import concat
from .utils import consume
from .utils import import_string
from .utils import internalcode
from .utils import LRUCache
from .utils import missing
def compile_templates(self, target: t.Union[str, os.PathLike], extensions: t.Optional[t.Collection[str]]=None, filter_func: t.Optional[t.Callable[[str], bool]]=None, zip: t.Optional[str]='deflated', log_function: t.Optional[t.Callable[[str], None]]=None, ignore_errors: bool=True) -> None:
    """Finds all the templates the loader can find, compiles them
        and stores them in `target`.  If `zip` is `None`, instead of in a
        zipfile, the templates will be stored in a directory.
        By default a deflate zip algorithm is used. To switch to
        the stored algorithm, `zip` can be set to ``'stored'``.

        `extensions` and `filter_func` are passed to :meth:`list_templates`.
        Each template returned will be compiled to the target folder or
        zipfile.

        By default template compilation errors are ignored.  In case a
        log function is provided, errors are logged.  If you want template
        syntax errors to abort the compilation you can set `ignore_errors`
        to `False` and you will get an exception on syntax errors.

        .. versionadded:: 2.4
        """
    from .loaders import ModuleLoader
    if log_function is None:

        def log_function(x: str) -> None:
            pass
    assert log_function is not None
    assert self.loader is not None, 'No loader configured.'

    def write_file(filename: str, data: str) -> None:
        if zip:
            info = ZipInfo(filename)
            info.external_attr = 493 << 16
            zip_file.writestr(info, data)
        else:
            with open(os.path.join(target, filename), 'wb') as f:
                f.write(data.encode('utf8'))
    if zip is not None:
        from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED
        zip_file = ZipFile(target, 'w', dict(deflated=ZIP_DEFLATED, stored=ZIP_STORED)[zip])
        log_function(f'Compiling into Zip archive {target!r}')
    else:
        if not os.path.isdir(target):
            os.makedirs(target)
        log_function(f'Compiling into folder {target!r}')
    try:
        for name in self.list_templates(extensions, filter_func):
            source, filename, _ = self.loader.get_source(self, name)
            try:
                code = self.compile(source, name, filename, True, True)
            except TemplateSyntaxError as e:
                if not ignore_errors:
                    raise
                log_function(f'Could not compile "{name}": {e}')
                continue
            filename = ModuleLoader.get_module_filename(name)
            write_file(filename, code)
            log_function(f'Compiled "{name}" as {filename}')
    finally:
        if zip:
            zip_file.close()
    log_function('Finished compiling templates')