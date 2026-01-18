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
def relationship(argument: Optional[_RelationshipArgumentType[Any]]=None, secondary: Optional[_RelationshipSecondaryArgument]=None, *, uselist: Optional[bool]=None, collection_class: Optional[Union[Type[Collection[Any]], Callable[[], Collection[Any]]]]=None, primaryjoin: Optional[_RelationshipJoinConditionArgument]=None, secondaryjoin: Optional[_RelationshipJoinConditionArgument]=None, back_populates: Optional[str]=None, order_by: _ORMOrderByArgument=False, backref: Optional[ORMBackrefArgument]=None, overlaps: Optional[str]=None, post_update: bool=False, cascade: str='save-update, merge', viewonly: bool=False, init: Union[_NoArg, bool]=_NoArg.NO_ARG, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, default: Union[_NoArg, _T]=_NoArg.NO_ARG, default_factory: Union[_NoArg, Callable[[], _T]]=_NoArg.NO_ARG, compare: Union[_NoArg, bool]=_NoArg.NO_ARG, kw_only: Union[_NoArg, bool]=_NoArg.NO_ARG, lazy: _LazyLoadArgumentType='select', passive_deletes: Union[Literal['all'], bool]=False, passive_updates: bool=True, active_history: bool=False, enable_typechecks: bool=True, foreign_keys: Optional[_ORMColCollectionArgument]=None, remote_side: Optional[_ORMColCollectionArgument]=None, join_depth: Optional[int]=None, comparator_factory: Optional[Type[RelationshipProperty.Comparator[Any]]]=None, single_parent: bool=False, innerjoin: bool=False, distinct_target_key: Optional[bool]=None, load_on_pending: bool=False, query_class: Optional[Type[Query[Any]]]=None, info: Optional[_InfoType]=None, omit_join: Literal[None, False]=None, sync_backref: Optional[bool]=None, **kw: Any) -> _RelationshipDeclared[Any]:
    """Provide a relationship between two mapped classes.

    This corresponds to a parent-child or associative table relationship.
    The constructed class is an instance of :class:`.Relationship`.

    .. seealso::

        :ref:`tutorial_orm_related_objects` - tutorial introduction
        to :func:`_orm.relationship` in the :ref:`unified_tutorial`

        :ref:`relationship_config_toplevel` - narrative documentation

    :param argument:
      This parameter refers to the class that is to be related.   It
      accepts several forms, including a direct reference to the target
      class itself, the :class:`_orm.Mapper` instance for the target class,
      a Python callable / lambda that will return a reference to the
      class or :class:`_orm.Mapper` when called, and finally a string
      name for the class, which will be resolved from the
      :class:`_orm.registry` in use in order to locate the class, e.g.::

            class SomeClass(Base):
                # ...

                related = relationship("RelatedClass")

      The :paramref:`_orm.relationship.argument` may also be omitted from the
      :func:`_orm.relationship` construct entirely, and instead placed inside
      a :class:`_orm.Mapped` annotation on the left side, which should
      include a Python collection type if the relationship is expected
      to be a collection, such as::

            class SomeClass(Base):
                # ...

                related_items: Mapped[List["RelatedItem"]] = relationship()

      Or for a many-to-one or one-to-one relationship::

            class SomeClass(Base):
                # ...

                related_item: Mapped["RelatedItem"] = relationship()

      .. seealso::

        :ref:`orm_declarative_properties` - further detail
        on relationship configuration when using Declarative.

    :param secondary:
      For a many-to-many relationship, specifies the intermediary
      table, and is typically an instance of :class:`_schema.Table`.
      In less common circumstances, the argument may also be specified
      as an :class:`_expression.Alias` construct, or even a
      :class:`_expression.Join` construct.

      :paramref:`_orm.relationship.secondary` may
      also be passed as a callable function which is evaluated at
      mapper initialization time.  When using Declarative, it may also
      be a string argument noting the name of a :class:`_schema.Table`
      that is
      present in the :class:`_schema.MetaData`
      collection associated with the
      parent-mapped :class:`_schema.Table`.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      The :paramref:`_orm.relationship.secondary` keyword argument is
      typically applied in the case where the intermediary
      :class:`_schema.Table`
      is not otherwise expressed in any direct class mapping. If the
      "secondary" table is also explicitly mapped elsewhere (e.g. as in
      :ref:`association_pattern`), one should consider applying the
      :paramref:`_orm.relationship.viewonly` flag so that this
      :func:`_orm.relationship`
      is not used for persistence operations which
      may conflict with those of the association object pattern.

      .. seealso::

          :ref:`relationships_many_to_many` - Reference example of "many
          to many".

          :ref:`self_referential_many_to_many` - Specifics on using
          many-to-many in a self-referential case.

          :ref:`declarative_many_to_many` - Additional options when using
          Declarative.

          :ref:`association_pattern` - an alternative to
          :paramref:`_orm.relationship.secondary`
          when composing association
          table relationships, allowing additional attributes to be
          specified on the association table.

          :ref:`composite_secondary_join` - a lesser-used pattern which
          in some cases can enable complex :func:`_orm.relationship` SQL
          conditions to be used.

    :param active_history=False:
      When ``True``, indicates that the "previous" value for a
      many-to-one reference should be loaded when replaced, if
      not already loaded. Normally, history tracking logic for
      simple many-to-ones only needs to be aware of the "new"
      value in order to perform a flush. This flag is available
      for applications that make use of
      :func:`.attributes.get_history` which also need to know
      the "previous" value of the attribute.

    :param backref:
      A reference to a string relationship name, or a :func:`_orm.backref`
      construct, which will be used to automatically generate a new
      :func:`_orm.relationship` on the related class, which then refers to this
      one using a bi-directional :paramref:`_orm.relationship.back_populates`
      configuration.

      In modern Python, explicit use of :func:`_orm.relationship`
      with :paramref:`_orm.relationship.back_populates` should be preferred,
      as it is more robust in terms of mapper configuration as well as
      more conceptually straightforward.  It also integrates with
      new :pep:`484` typing features introduced in SQLAlchemy 2.0 which
      is not possible with dynamically generated attributes.

      .. seealso::

        :ref:`relationships_backref` - notes on using
        :paramref:`_orm.relationship.backref`

        :ref:`tutorial_orm_related_objects` - in the :ref:`unified_tutorial`,
        presents an overview of bi-directional relationship configuration
        and behaviors using :paramref:`_orm.relationship.back_populates`

        :func:`.backref` - allows control over :func:`_orm.relationship`
        configuration when using :paramref:`_orm.relationship.backref`.


    :param back_populates:
      Indicates the name of a :func:`_orm.relationship` on the related
      class that will be synchronized with this one.   It is usually
      expected that the :func:`_orm.relationship` on the related class
      also refer to this one.  This allows objects on both sides of
      each :func:`_orm.relationship` to synchronize in-Python state
      changes and also provides directives to the :term:`unit of work`
      flush process how changes along these relationships should
      be persisted.

      .. seealso::

        :ref:`tutorial_orm_related_objects` - in the :ref:`unified_tutorial`,
        presents an overview of bi-directional relationship configuration
        and behaviors.

        :ref:`relationship_patterns` - includes many examples of
        :paramref:`_orm.relationship.back_populates`.

        :paramref:`_orm.relationship.backref` - legacy form which allows
        more succinct configuration, but does not support explicit typing

    :param overlaps:
       A string name or comma-delimited set of names of other relationships
       on either this mapper, a descendant mapper, or a target mapper with
       which this relationship may write to the same foreign keys upon
       persistence.   The only effect this has is to eliminate the
       warning that this relationship will conflict with another upon
       persistence.   This is used for such relationships that are truly
       capable of conflicting with each other on write, but the application
       will ensure that no such conflicts occur.

       .. versionadded:: 1.4

       .. seealso::

            :ref:`error_qzyx` - usage example

    :param cascade:
      A comma-separated list of cascade rules which determines how
      Session operations should be "cascaded" from parent to child.
      This defaults to ``False``, which means the default cascade
      should be used - this default cascade is ``"save-update, merge"``.

      The available cascades are ``save-update``, ``merge``,
      ``expunge``, ``delete``, ``delete-orphan``, and ``refresh-expire``.
      An additional option, ``all`` indicates shorthand for
      ``"save-update, merge, refresh-expire,
      expunge, delete"``, and is often used as in ``"all, delete-orphan"``
      to indicate that related objects should follow along with the
      parent object in all cases, and be deleted when de-associated.

      .. seealso::

        :ref:`unitofwork_cascades` - Full detail on each of the available
        cascade options.

    :param cascade_backrefs=False:
      Legacy; this flag is always False.

      .. versionchanged:: 2.0 "cascade_backrefs" functionality has been
         removed.

    :param collection_class:
      A class or callable that returns a new list-holding object. will
      be used in place of a plain list for storing elements.

      .. seealso::

        :ref:`custom_collections` - Introductory documentation and
        examples.

    :param comparator_factory:
      A class which extends :class:`.Relationship.Comparator`
      which provides custom SQL clause generation for comparison
      operations.

      .. seealso::

        :class:`.PropComparator` - some detail on redefining comparators
        at this level.

        :ref:`custom_comparators` - Brief intro to this feature.


    :param distinct_target_key=None:
      Indicate if a "subquery" eager load should apply the DISTINCT
      keyword to the innermost SELECT statement.  When left as ``None``,
      the DISTINCT keyword will be applied in those cases when the target
      columns do not comprise the full primary key of the target table.
      When set to ``True``, the DISTINCT keyword is applied to the
      innermost SELECT unconditionally.

      It may be desirable to set this flag to False when the DISTINCT is
      reducing performance of the innermost subquery beyond that of what
      duplicate innermost rows may be causing.

      .. seealso::

        :ref:`loading_toplevel` - includes an introduction to subquery
        eager loading.

    :param doc:
      Docstring which will be applied to the resulting descriptor.

    :param foreign_keys:

      A list of columns which are to be used as "foreign key"
      columns, or columns which refer to the value in a remote
      column, within the context of this :func:`_orm.relationship`
      object's :paramref:`_orm.relationship.primaryjoin` condition.
      That is, if the :paramref:`_orm.relationship.primaryjoin`
      condition of this :func:`_orm.relationship` is ``a.id ==
      b.a_id``, and the values in ``b.a_id`` are required to be
      present in ``a.id``, then the "foreign key" column of this
      :func:`_orm.relationship` is ``b.a_id``.

      In normal cases, the :paramref:`_orm.relationship.foreign_keys`
      parameter is **not required.** :func:`_orm.relationship` will
      automatically determine which columns in the
      :paramref:`_orm.relationship.primaryjoin` condition are to be
      considered "foreign key" columns based on those
      :class:`_schema.Column` objects that specify
      :class:`_schema.ForeignKey`,
      or are otherwise listed as referencing columns in a
      :class:`_schema.ForeignKeyConstraint` construct.
      :paramref:`_orm.relationship.foreign_keys` is only needed when:

        1. There is more than one way to construct a join from the local
           table to the remote table, as there are multiple foreign key
           references present.  Setting ``foreign_keys`` will limit the
           :func:`_orm.relationship`
           to consider just those columns specified
           here as "foreign".

        2. The :class:`_schema.Table` being mapped does not actually have
           :class:`_schema.ForeignKey` or
           :class:`_schema.ForeignKeyConstraint`
           constructs present, often because the table
           was reflected from a database that does not support foreign key
           reflection (MySQL MyISAM).

        3. The :paramref:`_orm.relationship.primaryjoin`
           argument is used to
           construct a non-standard join condition, which makes use of
           columns or expressions that do not normally refer to their
           "parent" column, such as a join condition expressed by a
           complex comparison using a SQL function.

      The :func:`_orm.relationship` construct will raise informative
      error messages that suggest the use of the
      :paramref:`_orm.relationship.foreign_keys` parameter when
      presented with an ambiguous condition.   In typical cases,
      if :func:`_orm.relationship` doesn't raise any exceptions, the
      :paramref:`_orm.relationship.foreign_keys` parameter is usually
      not needed.

      :paramref:`_orm.relationship.foreign_keys` may also be passed as a
      callable function which is evaluated at mapper initialization time,
      and may be passed as a Python-evaluable string when using
      Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      .. seealso::

        :ref:`relationship_foreign_keys`

        :ref:`relationship_custom_foreign`

        :func:`.foreign` - allows direct annotation of the "foreign"
        columns within a :paramref:`_orm.relationship.primaryjoin`
        condition.

    :param info: Optional data dictionary which will be populated into the
        :attr:`.MapperProperty.info` attribute of this object.

    :param innerjoin=False:
      When ``True``, joined eager loads will use an inner join to join
      against related tables instead of an outer join.  The purpose
      of this option is generally one of performance, as inner joins
      generally perform better than outer joins.

      This flag can be set to ``True`` when the relationship references an
      object via many-to-one using local foreign keys that are not
      nullable, or when the reference is one-to-one or a collection that
      is guaranteed to have one or at least one entry.

      The option supports the same "nested" and "unnested" options as
      that of :paramref:`_orm.joinedload.innerjoin`.  See that flag
      for details on nested / unnested behaviors.

      .. seealso::

        :paramref:`_orm.joinedload.innerjoin` - the option as specified by
        loader option, including detail on nesting behavior.

        :ref:`what_kind_of_loading` - Discussion of some details of
        various loader options.


    :param join_depth:
      When non-``None``, an integer value indicating how many levels
      deep "eager" loaders should join on a self-referring or cyclical
      relationship.  The number counts how many times the same Mapper
      shall be present in the loading condition along a particular join
      branch.  When left at its default of ``None``, eager loaders
      will stop chaining when they encounter a the same target mapper
      which is already higher up in the chain.  This option applies
      both to joined- and subquery- eager loaders.

      .. seealso::

        :ref:`self_referential_eager_loading` - Introductory documentation
        and examples.

    :param lazy='select': specifies
      How the related items should be loaded.  Default value is
      ``select``.  Values include:

      * ``select`` - items should be loaded lazily when the property is
        first accessed, using a separate SELECT statement, or identity map
        fetch for simple many-to-one references.

      * ``immediate`` - items should be loaded as the parents are loaded,
        using a separate SELECT statement, or identity map fetch for
        simple many-to-one references.

      * ``joined`` - items should be loaded "eagerly" in the same query as
        that of the parent, using a JOIN or LEFT OUTER JOIN.  Whether
        the join is "outer" or not is determined by the
        :paramref:`_orm.relationship.innerjoin` parameter.

      * ``subquery`` - items should be loaded "eagerly" as the parents are
        loaded, using one additional SQL statement, which issues a JOIN to
        a subquery of the original statement, for each collection
        requested.

      * ``selectin`` - items should be loaded "eagerly" as the parents
        are loaded, using one or more additional SQL statements, which
        issues a JOIN to the immediate parent object, specifying primary
        key identifiers using an IN clause.

      * ``noload`` - no loading should occur at any time.  The related
        collection will remain empty.   The ``noload`` strategy is not
        recommended for general use.  For a general use "never load"
        approach, see :ref:`write_only_relationship`

      * ``raise`` - lazy loading is disallowed; accessing
        the attribute, if its value were not already loaded via eager
        loading, will raise an :exc:`~sqlalchemy.exc.InvalidRequestError`.
        This strategy can be used when objects are to be detached from
        their attached :class:`.Session` after they are loaded.

      * ``raise_on_sql`` - lazy loading that emits SQL is disallowed;
        accessing the attribute, if its value were not already loaded via
        eager loading, will raise an
        :exc:`~sqlalchemy.exc.InvalidRequestError`, **if the lazy load
        needs to emit SQL**.  If the lazy load can pull the related value
        from the identity map or determine that it should be None, the
        value is loaded.  This strategy can be used when objects will
        remain associated with the attached :class:`.Session`, however
        additional SELECT statements should be blocked.

      * ``write_only`` - the attribute will be configured with a special
        "virtual collection" that may receive
        :meth:`_orm.WriteOnlyCollection.add` and
        :meth:`_orm.WriteOnlyCollection.remove` commands to add or remove
        individual objects, but will not under any circumstances load or
        iterate the full set of objects from the database directly. Instead,
        methods such as :meth:`_orm.WriteOnlyCollection.select`,
        :meth:`_orm.WriteOnlyCollection.insert`,
        :meth:`_orm.WriteOnlyCollection.update` and
        :meth:`_orm.WriteOnlyCollection.delete` are provided which generate SQL
        constructs that may be used to load and modify rows in bulk. Used for
        large collections that are never appropriate to load at once into
        memory.

        The ``write_only`` loader style is configured automatically when
        the :class:`_orm.WriteOnlyMapped` annotation is provided on the
        left hand side within a Declarative mapping.  See the section
        :ref:`write_only_relationship` for examples.

        .. versionadded:: 2.0

        .. seealso::

            :ref:`write_only_relationship` - in the :ref:`queryguide_toplevel`

      * ``dynamic`` - the attribute will return a pre-configured
        :class:`_query.Query` object for all read
        operations, onto which further filtering operations can be
        applied before iterating the results.

        The ``dynamic`` loader style is configured automatically when
        the :class:`_orm.DynamicMapped` annotation is provided on the
        left hand side within a Declarative mapping.  See the section
        :ref:`dynamic_relationship` for examples.

        .. legacy::  The "dynamic" lazy loader strategy is the legacy form of
           what is now the "write_only" strategy described in the section
           :ref:`write_only_relationship`.

        .. seealso::

            :ref:`dynamic_relationship` - in the :ref:`queryguide_toplevel`

            :ref:`write_only_relationship` - more generally useful approach
            for large collections that should not fully load into memory

      * True - a synonym for 'select'

      * False - a synonym for 'joined'

      * None - a synonym for 'noload'

      .. seealso::

        :ref:`orm_queryguide_relationship_loaders` - Full documentation on
        relationship loader configuration in the :ref:`queryguide_toplevel`.


    :param load_on_pending=False:
      Indicates loading behavior for transient or pending parent objects.

      When set to ``True``, causes the lazy-loader to
      issue a query for a parent object that is not persistent, meaning it
      has never been flushed.  This may take effect for a pending object
      when autoflush is disabled, or for a transient object that has been
      "attached" to a :class:`.Session` but is not part of its pending
      collection.

      The :paramref:`_orm.relationship.load_on_pending`
      flag does not improve
      behavior when the ORM is used normally - object references should be
      constructed at the object level, not at the foreign key level, so
      that they are present in an ordinary way before a flush proceeds.
      This flag is not not intended for general use.

      .. seealso::

          :meth:`.Session.enable_relationship_loading` - this method
          establishes "load on pending" behavior for the whole object, and
          also allows loading on objects that remain transient or
          detached.

    :param order_by:
      Indicates the ordering that should be applied when loading these
      items.  :paramref:`_orm.relationship.order_by`
      is expected to refer to
      one of the :class:`_schema.Column`
      objects to which the target class is
      mapped, or the attribute itself bound to the target class which
      refers to the column.

      :paramref:`_orm.relationship.order_by`
      may also be passed as a callable
      function which is evaluated at mapper initialization time, and may
      be passed as a Python-evaluable string when using Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

    :param passive_deletes=False:
       Indicates loading behavior during delete operations.

       A value of True indicates that unloaded child items should not
       be loaded during a delete operation on the parent.  Normally,
       when a parent item is deleted, all child items are loaded so
       that they can either be marked as deleted, or have their
       foreign key to the parent set to NULL.  Marking this flag as
       True usually implies an ON DELETE <CASCADE|SET NULL> rule is in
       place which will handle updating/deleting child rows on the
       database side.

       Additionally, setting the flag to the string value 'all' will
       disable the "nulling out" of the child foreign keys, when the parent
       object is deleted and there is no delete or delete-orphan cascade
       enabled.  This is typically used when a triggering or error raise
       scenario is in place on the database side.  Note that the foreign
       key attributes on in-session child objects will not be changed after
       a flush occurs so this is a very special use-case setting.
       Additionally, the "nulling out" will still occur if the child
       object is de-associated with the parent.

       .. seealso::

            :ref:`passive_deletes` - Introductory documentation
            and examples.

    :param passive_updates=True:
      Indicates the persistence behavior to take when a referenced
      primary key value changes in place, indicating that the referencing
      foreign key columns will also need their value changed.

      When True, it is assumed that ``ON UPDATE CASCADE`` is configured on
      the foreign key in the database, and that the database will
      handle propagation of an UPDATE from a source column to
      dependent rows.  When False, the SQLAlchemy
      :func:`_orm.relationship`
      construct will attempt to emit its own UPDATE statements to
      modify related targets.  However note that SQLAlchemy **cannot**
      emit an UPDATE for more than one level of cascade.  Also,
      setting this flag to False is not compatible in the case where
      the database is in fact enforcing referential integrity, unless
      those constraints are explicitly "deferred", if the target backend
      supports it.

      It is highly advised that an application which is employing
      mutable primary keys keeps ``passive_updates`` set to True,
      and instead uses the referential integrity features of the database
      itself in order to handle the change efficiently and fully.

      .. seealso::

          :ref:`passive_updates` - Introductory documentation and
          examples.

          :paramref:`.mapper.passive_updates` - a similar flag which
          takes effect for joined-table inheritance mappings.

    :param post_update:
      This indicates that the relationship should be handled by a
      second UPDATE statement after an INSERT or before a
      DELETE. This flag is used to handle saving bi-directional
      dependencies between two individual rows (i.e. each row
      references the other), where it would otherwise be impossible to
      INSERT or DELETE both rows fully since one row exists before the
      other. Use this flag when a particular mapping arrangement will
      incur two rows that are dependent on each other, such as a table
      that has a one-to-many relationship to a set of child rows, and
      also has a column that references a single child row within that
      list (i.e. both tables contain a foreign key to each other). If
      a flush operation returns an error that a "cyclical
      dependency" was detected, this is a cue that you might want to
      use :paramref:`_orm.relationship.post_update` to "break" the cycle.

      .. seealso::

          :ref:`post_update` - Introductory documentation and examples.

    :param primaryjoin:
      A SQL expression that will be used as the primary
      join of the child object against the parent object, or in a
      many-to-many relationship the join of the parent object to the
      association table. By default, this value is computed based on the
      foreign key relationships of the parent and child tables (or
      association table).

      :paramref:`_orm.relationship.primaryjoin` may also be passed as a
      callable function which is evaluated at mapper initialization time,
      and may be passed as a Python-evaluable string when using
      Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      .. seealso::

          :ref:`relationship_primaryjoin`

    :param remote_side:
      Used for self-referential relationships, indicates the column or
      list of columns that form the "remote side" of the relationship.

      :paramref:`_orm.relationship.remote_side` may also be passed as a
      callable function which is evaluated at mapper initialization time,
      and may be passed as a Python-evaluable string when using
      Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      .. seealso::

        :ref:`self_referential` - in-depth explanation of how
        :paramref:`_orm.relationship.remote_side`
        is used to configure self-referential relationships.

        :func:`.remote` - an annotation function that accomplishes the
        same purpose as :paramref:`_orm.relationship.remote_side`,
        typically
        when a custom :paramref:`_orm.relationship.primaryjoin` condition
        is used.

    :param query_class:
      A :class:`_query.Query`
      subclass that will be used internally by the
      ``AppenderQuery`` returned by a "dynamic" relationship, that
      is, a relationship that specifies ``lazy="dynamic"`` or was
      otherwise constructed using the :func:`_orm.dynamic_loader`
      function.

      .. seealso::

        :ref:`dynamic_relationship` - Introduction to "dynamic"
        relationship loaders.

    :param secondaryjoin:
      A SQL expression that will be used as the join of
      an association table to the child object. By default, this value is
      computed based on the foreign key relationships of the association
      and child tables.

      :paramref:`_orm.relationship.secondaryjoin` may also be passed as a
      callable function which is evaluated at mapper initialization time,
      and may be passed as a Python-evaluable string when using
      Declarative.

      .. warning:: When passed as a Python-evaluable string, the
         argument is interpreted using Python's ``eval()`` function.
         **DO NOT PASS UNTRUSTED INPUT TO THIS STRING**.
         See :ref:`declarative_relationship_eval` for details on
         declarative evaluation of :func:`_orm.relationship` arguments.

      .. seealso::

          :ref:`relationship_primaryjoin`

    :param single_parent:
      When True, installs a validator which will prevent objects
      from being associated with more than one parent at a time.
      This is used for many-to-one or many-to-many relationships that
      should be treated either as one-to-one or one-to-many.  Its usage
      is optional, except for :func:`_orm.relationship` constructs which
      are many-to-one or many-to-many and also
      specify the ``delete-orphan`` cascade option.  The
      :func:`_orm.relationship` construct itself will raise an error
      instructing when this option is required.

      .. seealso::

        :ref:`unitofwork_cascades` - includes detail on when the
        :paramref:`_orm.relationship.single_parent`
        flag may be appropriate.

    :param uselist:
      A boolean that indicates if this property should be loaded as a
      list or a scalar. In most cases, this value is determined
      automatically by :func:`_orm.relationship` at mapper configuration
      time.  When using explicit :class:`_orm.Mapped` annotations,
      :paramref:`_orm.relationship.uselist` may be derived from the
      whether or not the annotation within :class:`_orm.Mapped` contains
      a collection class.
      Otherwise, :paramref:`_orm.relationship.uselist` may be derived from
      the type and direction
      of the relationship - one to many forms a list, many to one
      forms a scalar, many to many is a list. If a scalar is desired
      where normally a list would be present, such as a bi-directional
      one-to-one relationship, use an appropriate :class:`_orm.Mapped`
      annotation or set :paramref:`_orm.relationship.uselist` to False.

      The :paramref:`_orm.relationship.uselist`
      flag is also available on an
      existing :func:`_orm.relationship`
      construct as a read-only attribute,
      which can be used to determine if this :func:`_orm.relationship`
      deals
      with collections or scalar attributes::

          >>> User.addresses.property.uselist
          True

      .. seealso::

          :ref:`relationships_one_to_one` - Introduction to the "one to
          one" relationship pattern, which is typically when an alternate
          setting for :paramref:`_orm.relationship.uselist` is involved.

    :param viewonly=False:
      When set to ``True``, the relationship is used only for loading
      objects, and not for any persistence operation.  A
      :func:`_orm.relationship` which specifies
      :paramref:`_orm.relationship.viewonly` can work
      with a wider range of SQL operations within the
      :paramref:`_orm.relationship.primaryjoin` condition, including
      operations that feature the use of a variety of comparison operators
      as well as SQL functions such as :func:`_expression.cast`.  The
      :paramref:`_orm.relationship.viewonly`
      flag is also of general use when defining any kind of
      :func:`_orm.relationship` that doesn't represent
      the full set of related objects, to prevent modifications of the
      collection from resulting in persistence operations.

      .. seealso::

        :ref:`relationship_viewonly_notes` - more details on best practices
        when using :paramref:`_orm.relationship.viewonly`.

    :param sync_backref:
      A boolean that enables the events used to synchronize the in-Python
      attributes when this relationship is target of either
      :paramref:`_orm.relationship.backref` or
      :paramref:`_orm.relationship.back_populates`.

      Defaults to ``None``, which indicates that an automatic value should
      be selected based on the value of the
      :paramref:`_orm.relationship.viewonly` flag.  When left at its
      default, changes in state will be back-populated only if neither
      sides of a relationship is viewonly.

      .. versionadded:: 1.3.17

      .. versionchanged:: 1.4 - A relationship that specifies
         :paramref:`_orm.relationship.viewonly` automatically implies
         that :paramref:`_orm.relationship.sync_backref` is ``False``.

      .. seealso::

        :paramref:`_orm.relationship.viewonly`

    :param omit_join:
      Allows manual control over the "selectin" automatic join
      optimization.  Set to ``False`` to disable the "omit join" feature
      added in SQLAlchemy 1.3; or leave as ``None`` to leave automatic
      optimization in place.

      .. note:: This flag may only be set to ``False``.   It is not
         necessary to set it to ``True`` as the "omit_join" optimization is
         automatically detected; if it is not detected, then the
         optimization is not supported.

         .. versionchanged:: 1.3.11  setting ``omit_join`` to True will now
            emit a warning as this was not the intended use of this flag.

      .. versionadded:: 1.3

    :param init: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__init__()``
     method as generated by the dataclass process.
    :param repr: Specific to :ref:`orm_declarative_native_dataclasses`,
     specifies if the mapped attribute should be part of the ``__repr__()``
     method as generated by the dataclass process.
    :param default_factory: Specific to
     :ref:`orm_declarative_native_dataclasses`,
     specifies a default-value generation function that will take place
     as part of the ``__init__()``
     method as generated by the dataclass process.
    :param compare: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be included in comparison operations when generating the
     ``__eq__()`` and ``__ne__()`` methods for the mapped class.

     .. versionadded:: 2.0.0b4

    :param kw_only: Specific to
     :ref:`orm_declarative_native_dataclasses`, indicates if this field
     should be marked as keyword-only when generating the ``__init__()``.


    """
    return _RelationshipDeclared(argument, secondary=secondary, uselist=uselist, collection_class=collection_class, primaryjoin=primaryjoin, secondaryjoin=secondaryjoin, back_populates=back_populates, order_by=order_by, backref=backref, overlaps=overlaps, post_update=post_update, cascade=cascade, viewonly=viewonly, attribute_options=_AttributeOptions(init, repr, default, default_factory, compare, kw_only), lazy=lazy, passive_deletes=passive_deletes, passive_updates=passive_updates, active_history=active_history, enable_typechecks=enable_typechecks, foreign_keys=foreign_keys, remote_side=remote_side, join_depth=join_depth, comparator_factory=comparator_factory, single_parent=single_parent, innerjoin=innerjoin, distinct_target_key=distinct_target_key, load_on_pending=load_on_pending, query_class=query_class, info=info, omit_join=omit_join, sync_backref=sync_backref, **kw)