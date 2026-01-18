from typing import TYPE_CHECKING, Any, Callable, List, TypeVar
import ray
from ray.util.annotations import DeveloperAPI
def map_unordered(self, fn: Callable[['ray.actor.ActorHandle', V], Any], values: List[V]):
    """Similar to map(), but returning an unordered iterator.

        This returns an unordered iterator that will return results of the map
        as they finish. This can be more efficient that map() if some results
        take longer to compute than others.

        Arguments:
            fn: Function that takes (actor, value) as argument and
                returns an ObjectRef computing the result over the value. The
                actor will be considered busy until the ObjectRef completes.
            values: List of values that fn(actor, value) should be
                applied to.

        Returns:
            Iterator over results from applying fn to the actors and values.

        Examples:
            .. testcode::

                import ray
                from ray.util.actor_pool import ActorPool

                @ray.remote
                class Actor:
                    def double(self, v):
                        return 2 * v

                a1, a2 = Actor.remote(), Actor.remote()
                pool = ActorPool([a1, a2])
                print(list(pool.map_unordered(lambda a, v: a.double.remote(v),
                                              [1, 2, 3, 4])))

            .. testoutput::
                :options: +MOCK

                [6, 8, 4, 2]
        """
    while self.has_next():
        try:
            self.get_next_unordered(timeout=0)
        except TimeoutError:
            pass
    for v in values:
        self.submit(fn, v)

    def get_generator():
        while self.has_next():
            yield self.get_next_unordered()
    return get_generator()