import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
@wrappers.Request.application
def serve_source_file(self, request):
    """Serves the content of a given source file.

        The source file is referred to by the index in the list of all source
        files involved in the execution of the debugged program, which is
        available via the `serve_source_files_list()`  serving route.

        Args:
          request: HTTP request.

        Returns:
          Response to the request.
        """
    experiment = plugin_util.experiment_id(request.environ)
    run = request.args.get('run')
    if run is None:
        return _missing_run_error_response(request)
    index = request.args.get('index')
    if index is None:
        return _error_response(request, 'index is not provided for source file content')
    index = int(index)
    run_tag_filter = debug_data_provider.source_file_run_tag_filter(run, index)
    blob_sequences = self._data_provider.read_blob_sequences(experiment_id=experiment, plugin_name=self.plugin_name, run_tag_filter=run_tag_filter)
    tag = next(iter(run_tag_filter.tags))
    try:
        return http_util.Respond(request, self._data_provider.read_blob(blob_key=blob_sequences[run][tag][0].blob_key), 'application/json')
    except errors.NotFoundError as e:
        return _error_response(request, str(e))