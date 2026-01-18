import json
import os
import sys
def send_runtime_params(params, appinfo=None):
    """Send runtime parameters back to the controller.

    Args:
        params: ({str: object, ...}) Set of runtime parameters.  Must be
            json-encodable.
        appinfo: ({str: object, ...} or None) Contents of the app.yaml file to
            be produced by the runtime definition.  Required fields may be
            added to this by the framework, the only thing an application
            needs to provide is the "runtime" field and any additional data
            fields.
    """
    if appinfo is not None:
        _write_msg(type='runtime_parameters', runtime_data=params, appinfo=appinfo)
    else:
        _write_msg(type='runtime_parameters', runtime_data=params)