from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import mapperlib as mapperlib
from ._typing import _O
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .interfaces import _AttributeOptions
from .properties import MappedColumn
from .properties import MappedSQLExpression
from .query import AliasOption
from .relationships import _RelationshipArgumentType
from .relationships import _RelationshipDeclared
from .relationships import _RelationshipSecondaryArgument
from .relationships import RelationshipProperty
from .session import Session
from .util import _ORMJoin
from .util import AliasedClass
from .util import AliasedInsp
from .util import LoaderCriteriaOption
from .. import sql
from .. import util
from ..exc import InvalidRequestError
from ..sql._typing import _no_kw
from ..sql.base import _NoArg
from ..sql.base import SchemaEventTarget
from ..sql.schema import _InsertSentinelColumnDefault
from ..sql.schema import SchemaConst
from ..sql.selectable import FromClause
from ..util.typing import Annotated
from ..util.typing import Literal
def synonym(name: str, *, map_column: Optional[bool]=None, descriptor: Optional[Any]=None, comparator_factory: Optional[Type[PropComparator[_T]]]=None, init: Union[_NoArg, bool]=_NoArg.NO_ARG, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, default: Union[_NoArg, _T]=_NoArg.NO_ARG, default_factory: Union[_NoArg, Callable[[], _T]]=_NoArg.NO_ARG, compare: Union[_NoArg, bool]=_NoArg.NO_ARG, kw_only: Union[_NoArg, bool]=_NoArg.NO_ARG, info: Optional[_InfoType]=None, doc: Optional[str]=None) -> Synonym[Any]:
    """Denote an attribute name as a synonym to a mapped property,
    in that the attribute will mirror the value and expression behavior
    of another attribute.

    e.g.::

        class MyClass(Base):
            __tablename__ = 'my_table'

            id = Column(Integer, primary_key=True)
            job_status = Column(String(50))

            status = synonym("job_status")


    :param name: the name of the existing mapped property.  This
      can refer to the string name ORM-mapped attribute
      configured on the class, including column-bound attributes
      and relationships.

    :param descriptor: a Python :term:`descriptor` that will be used
      as a getter (and potentially a setter) when this attribute is
      accessed at the instance level.

    :param map_column: **For classical mappings and mappings against
      an existing Table object only**.  if ``True``, the :func:`.synonym`
      construct will locate the :class:`_schema.Column`
      object upon the mapped
      table that would normally be associated with the attribute name of
      this synonym, and produce a new :class:`.ColumnProperty` that instead
      maps this :class:`_schema.Column`
      to the alternate name given as the "name"
      argument of the synonym; in this way, the usual step of redefining
      the mapping of the :class:`_schema.Column`
      to be under a different name is
      unnecessary. This is usually intended to be used when a
      :class:`_schema.Column`
      is to be replaced with an attribute that also uses a
      descriptor, that is, in conjunction with the
      :paramref:`.synonym.descriptor` parameter::

        my_table = Table(
            "my_table", metadata,
            Column('id', Integer, primary_key=True),
            Column('job_status', String(50))
        )

        class MyClass:
            @property
            def _job_status_descriptor(self):
                return "Status: %s" % self._job_status


        mapper(
            MyClass, my_table, properties={
                "job_status": synonym(
                    "_job_status", map_column=True,
                    descriptor=MyClass._job_status_descriptor)
            }
        )

      Above, the attribute named ``_job_status`` is automatically
      mapped to the ``job_status`` column::

        >>> j1 = MyClass()
        >>> j1._job_status = "employed"
        >>> j1.job_status
        Status: employed

      When using Declarative, in order to provide a descriptor in
      conjunction with a synonym, use the
      :func:`sqlalchemy.ext.declarative.synonym_for` helper.  However,
      note that the :ref:`hybrid properties <mapper_hybrids>` feature
      should usually be preferred, particularly when redefining attribute
      behavior.

    :param info: Optional data dictionary which will be populated into the
        :attr:`.InspectionAttr.info` attribute of this object.

    :param comparator_factory: A subclass of :class:`.PropComparator`
      that will provide custom comparison behavior at the SQL expression
      level.

      .. note::

        For the use case of providing an attribute which redefines both
        Python-level and SQL-expression level behavior of an attribute,
        please refer to the Hybrid attribute introduced at
        :ref:`mapper_hybrids` for a more effective technique.

    .. seealso::

        :ref:`synonyms` - Overview of synonyms

        :func:`.synonym_for` - a helper oriented towards Declarative

        :ref:`mapper_hybrids` - The Hybrid Attribute extension provides an
        updated approach to augmenting attribute behavior more flexibly
        than can be achieved with synonyms.

    """
    return Synonym(name, map_column=map_column, descriptor=descriptor, comparator_factory=comparator_factory, attribute_options=_AttributeOptions(init, repr, default, default_factory, compare, kw_only), doc=doc, info=info)