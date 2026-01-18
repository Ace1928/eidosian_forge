from typedapi import ensure_api_is_typed
import autokeras
def test_api_surface_is_typed():
    ensure_api_is_typed([autokeras], EXCEPTION_LIST, init_only=True, additional_message=HELP_MESSAGE)