from typing import Optional, Union
def record_deprecated_state_api_import():
    import warnings
    from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
    warnings.warn('Ray state API is no longer experimental. Please import from `ray.util.state`. instead. Importing from `ray.experimental` will be deprecated in future releases. ', DeprecationWarning)
    record_extra_usage_tag(TagKey.EXPERIMENTAL_STATE_API_IMPORT, '1')