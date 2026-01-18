from typing import Any, Callable
from starlette.background import BackgroundTasks as StarletteBackgroundTasks
from typing_extensions import Annotated, Doc, ParamSpec  # type: ignore [attr-defined]
def add_task(self, func: Annotated[Callable[P, Any], Doc('\n                The function to call after the response is sent.\n\n                It can be a regular `def` function or an `async def` function.\n                ')], *args: P.args, **kwargs: P.kwargs) -> None:
    """
        Add a function to be called in the background after the response is sent.

        Read more about it in the
        [FastAPI docs for Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/).
        """
    return super().add_task(func, *args, **kwargs)