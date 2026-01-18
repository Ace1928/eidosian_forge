from lazyops.utils.imports import resolve_missing, require_missing_wrapper
def resolve_slack_sdk(required: bool=False):
    """
    Ensures that `slack_sdk` is available
    """
    global slack_sdk, _slack_sdk_available
    if not _slack_sdk_available:
        resolve_missing('slack_sdk', required=required)
        import slack_sdk
        _slack_sdk_available = True
        globals()['slack_sdk'] = slack_sdk