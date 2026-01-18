import collections
import functools
import itertools
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
def group_by_function(allow_aggregator_fallback):

    @specs.parameter('collection', yaqltypes.Iterable())
    @specs.parameter('key_selector', yaqltypes.Lambda())
    @specs.parameter('value_selector', yaqltypes.Lambda())
    @specs.parameter('aggregator', yaqltypes.Lambda())
    @specs.method
    def group_by(engine, collection, key_selector, value_selector=None, aggregator=None):
        """:yaql:groupBy

        Returns a collection grouped by keySelector with applied valueSelector
        as values. Returns a list of pairs where the first value is a result
        value of keySelector and the second is a list of values which have
        common keySelector return value.

        :signature: collection.groupBy(keySelector, valueSelector => null,
                                       aggregator => null)
        :receiverArg collection: input collection
        :argType collection: iterable
        :arg keySelector: function to be applied to every collection element.
            Values are grouped by return value of this function
        :argType keySelector: lambda
        :arg valueSelector: function to be applied to every collection element
            to put it under appropriate group. null by default, which means
            return element itself
        :argType valueSelector: lambda
        :arg aggregator: function to aggregate value within each group. null by
            default, which means no function to be evaluated on groups
        :argType aggregator: lambda
        :returnType: list

        .. code::

            yaql> [["a", 1], ["b", 2], ["c", 1], ["d", 2]].groupBy($[1], $[0])
            [[1, ["a", "c"]], [2, ["b", "d"]]]
            yaql> [["a", 1], ["b", 2], ["c", 1]].groupBy($[1], $[0], $.sum())
            [[1, "ac"], [2, "b"]]
        """
        groups = {}
        new_aggregator = GroupAggregator(aggregator, allow_aggregator_fallback)
        for t in collection:
            value = t if value_selector is None else value_selector(t)
            groups.setdefault(key_selector(t), []).append(value)
            utils.limit_memory_usage(engine, (1, groups))
        return select(groups.items(), new_aggregator)
    return group_by