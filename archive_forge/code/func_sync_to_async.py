import asyncio
import inspect
def sync_to_async(func):
    """Convert a blocking function to async function"""
    if is_async_func(func):
        return func

    async def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper