from sentry_sdk import consts
from sentry_sdk._types import TYPE_CHECKING
import sentry_sdk
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import logger, capture_internal_exceptions, event_from_exception
def new_iterator():
    with capture_internal_exceptions():
        for x in old_iterator:
            if hasattr(x, 'choices'):
                choice_index = 0
                for choice in x.choices:
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        content = choice.delta.content
                        if len(data_buf) <= choice_index:
                            data_buf.append([])
                        data_buf[choice_index].append(content or '')
                    choice_index += 1
            yield x
        if len(data_buf) > 0:
            all_responses = list(map(lambda chunk: ''.join(chunk), data_buf))
            if _should_send_default_pii() and integration.include_prompts:
                set_data_normalized(span, 'ai.responses', all_responses)
            _calculate_chat_completion_usage(messages, res, span, all_responses)
    span.__exit__(None, None, None)