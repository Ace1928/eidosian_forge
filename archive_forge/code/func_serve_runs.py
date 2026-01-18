import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
@wrappers.Request.application
def serve_runs(self, request):
    experiment = plugin_util.experiment_id(request.environ)
    runs = self._data_provider.list_runs(experiment_id=experiment)
    run_listing = dict()
    for run in runs:
        run_listing[run.run_id] = {'start_time': run.start_time}
    return http_util.Respond(request, run_listing, 'application/json')