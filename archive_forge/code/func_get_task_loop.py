import asyncio
import sys
def get_task_loop(task):
    return task._loop