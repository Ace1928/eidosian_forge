from typing import TYPE_CHECKING, Any, Callable, List, TypeVar
import ray
from ray.util.annotations import DeveloperAPI
def has_free(self):
    """Returns whether there are any idle actors available.

        Returns:
            True if there are any idle actors and no pending submits.

        Examples:
            .. testcode::

                import ray
                from ray.util.actor_pool import ActorPool

                @ray.remote
                class Actor:
                    def double(self, v):
                        return 2 * v

                a1 = Actor.remote()
                pool = ActorPool([a1])
                pool.submit(lambda a, v: a.double.remote(v), 1)
                print(pool.has_free())
                print(pool.get_next())
                print(pool.has_free())

            .. testoutput::

                False
                2
                True
        """
    return len(self._idle_actors) > 0 and len(self._pending_submits) == 0