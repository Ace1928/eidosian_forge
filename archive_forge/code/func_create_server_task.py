import time
import asyncio
import threading
from lazyops.utils.logs import logger
from typing import List, Union, Set, Callable, Tuple, Optional
def create_server_task(name: Optional[str]=None, primary_process_only: Optional[bool]=None, interval: Optional[float]=60.0, stop_on_error: Optional[bool]=True):
    """
    Create a server task and register it with this wrapper
    """

    def decorator(func):
        """
        Inner wrapper
        """

        async def wrapper_func(*args, **kwargs):
            iterations = 0
            while True:
                try:
                    await func(*args, iterations=iterations, **kwargs)
                    iterations += 1
                except Exception as e:
                    logger.trace(f'Error in server task {name or func.__name__}', e)
                    if stop_on_error:
                        logger.info(f'Stopping server task {name or func.__name__}...')
                        break
                await asyncio.sleep(interval)
        register_server_task(func=wrapper_func, name=name, primary_process_only=primary_process_only)
        return wrapper_func
    return decorator