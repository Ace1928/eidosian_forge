from functools import wraps
def branch_coro(self, fn):
    """
        Wraps a coroutine, and pass a new child callback to it.
        """

    @wraps(fn)
    async def func(path1, path2: str, **kwargs):
        with self.branched(path1, path2, **kwargs) as child:
            return await fn(path1, path2, callback=child, **kwargs)
    return func