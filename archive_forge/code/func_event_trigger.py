from __future__ import annotations
import dataclasses
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence
from gradio_client.documentation import document
from jinja2 import Template
from gradio.context import Context
from gradio.utils import get_cancel_function
def event_trigger(block: Block | None, fn: Callable | None | Literal['decorator']='decorator', inputs: Component | list[Component] | set[Component] | None=None, outputs: Component | list[Component] | None=None, api_name: str | None | Literal[False]=None, scroll_to_output: bool=False, show_progress: Literal['full', 'minimal', 'hidden']=_show_progress, queue: bool | None=None, batch: bool=False, max_batch_size: int=4, preprocess: bool=True, postprocess: bool=True, cancels: dict[str, Any] | list[dict[str, Any]] | None=None, every: float | None=None, trigger_mode: Literal['once', 'multiple', 'always_last'] | None=None, js: str | None=None, concurrency_limit: int | None | Literal['default']='default', concurrency_id: str | None=None, show_api: bool=True) -> Dependency:
    """
            Parameters:
                fn: the function to call when this event is triggered. Often a machine learning model's prediction function. Each parameter of the function corresponds to one input component, and the function should return a single value or a tuple of values, with each element in the tuple corresponding to one output component.
                inputs: List of gradio.components to use as inputs. If the function takes no inputs, this should be an empty list.
                outputs: List of gradio.components to use as outputs. If the function returns no outputs, this should be an empty list.
                api_name: defines how the endpoint appears in the API docs. Can be a string, None, or False. If set to a string, the endpoint will be exposed in the API docs with the given name. If None (default), the name of the function will be used as the API endpoint. If False, the endpoint will not be exposed in the API docs and downstream apps (including those that `gr.load` this app) will not be able to use this event.
                scroll_to_output: If True, will scroll to output component on completion
                show_progress: If True, will show progress animation while pending
                queue: If True, will place the request on the queue, if the queue has been enabled. If False, will not put this event on the queue, even if the queue has been enabled. If None, will use the queue setting of the gradio app.
                batch: If True, then the function should process a batch of inputs, meaning that it should accept a list of input values for each parameter. The lists should be of equal length (and be up to length `max_batch_size`). The function is then *required* to return a tuple of lists (even if there is only 1 output component), with each list in the tuple corresponding to one output component.
                max_batch_size: Maximum number of inputs to batch together if this is called from the queue (only relevant if batch=True)
                preprocess: If False, will not run preprocessing of component data before running 'fn' (e.g. leaving it as a base64 string if this method is called with the `Image` component).
                postprocess: If False, will not run postprocessing of component data before returning 'fn' output to the browser.
                cancels: A list of other events to cancel when this listener is triggered. For example, setting cancels=[click_event] will cancel the click_event, where click_event is the return value of another components .click method. Functions that have not yet run (or generators that are iterating) will be cancelled, but functions that are currently running will be allowed to finish.
                every: Run this event 'every' number of seconds while the client connection is open. Interpreted in seconds.
                trigger_mode: If "once" (default for all events except `.change()`) would not allow any submissions while an event is pending. If set to "multiple", unlimited submissions are allowed while pending, and "always_last" (default for `.change()` and `.key_up()` events) would allow a second submission after the pending event is complete.
                js: Optional frontend js method to run before running 'fn'. Input arguments for js method are values of 'inputs' and 'outputs', return should be a list of values for output components.
                concurrency_limit: If set, this is the maximum number of this event that can be running simultaneously. Can be set to None to mean no concurrency_limit (any number of this event can be running simultaneously). Set to "default" to use the default concurrency limit (defined by the `default_concurrency_limit` parameter in `Blocks.queue()`, which itself is 1 by default).
                concurrency_id: If set, this is the id of the concurrency group. Events with the same concurrency_id will be limited by the lowest set concurrency_limit.
                show_api: whether to show this event in the "view API" page of the Gradio app, or in the ".view_api()" method of the Gradio clients. Unlike setting api_name to False, setting show_api to False will still allow downstream apps to use this event. If fn is None, show_api will automatically be set to False.
            """
    if fn == 'decorator':

        def wrapper(func):
            event_trigger(block=block, fn=func, inputs=inputs, outputs=outputs, api_name=api_name, scroll_to_output=scroll_to_output, show_progress=show_progress, queue=queue, batch=batch, max_batch_size=max_batch_size, preprocess=preprocess, postprocess=postprocess, cancels=cancels, every=every, trigger_mode=trigger_mode, js=js, concurrency_limit=concurrency_limit, concurrency_id=concurrency_id, show_api=show_api)

            @wraps(func)
            def inner(*args, **kwargs):
                return func(*args, **kwargs)
            return inner
        return Dependency(None, {}, None, wrapper)
    from gradio.components.base import StreamingInput
    if isinstance(block, StreamingInput) and 'stream' in block.events:
        block.check_streamable()
    if isinstance(show_progress, bool):
        show_progress = 'full' if show_progress else 'hidden'
    if Context.root_block is None:
        raise AttributeError(f'Cannot call {_event_name} outside of a gradio.Blocks context.')
    dep, dep_index = Context.root_block.set_event_trigger([EventListenerMethod(block if _has_trigger else None, _event_name)], fn, inputs, outputs, preprocess=preprocess, postprocess=postprocess, scroll_to_output=scroll_to_output, show_progress=show_progress, api_name=api_name, js=js, concurrency_limit=concurrency_limit, concurrency_id=concurrency_id, queue=queue, batch=batch, max_batch_size=max_batch_size, every=every, trigger_after=_trigger_after, trigger_only_on_success=_trigger_only_on_success, trigger_mode=trigger_mode, show_api=show_api)
    set_cancel_events([EventListenerMethod(block if _has_trigger else None, _event_name)], cancels)
    if _callback:
        _callback(block)
    return Dependency(block, dep, dep_index, fn)