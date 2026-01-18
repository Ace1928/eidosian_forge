import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
@wrappers.Request.application
def serve_stack_frames(self, request):
    """Serves the content of stack frames.

        The source frames being requested are referred to be UUIDs for each of
        them, separated by commas.

        Args:
          request: HTTP request.

        Returns:
          Response to the request.
        """
    experiment = plugin_util.experiment_id(request.environ)
    run = request.args.get('run')
    if run is None:
        return _missing_run_error_response(request)
    stack_frame_ids = request.args.get('stack_frame_ids')
    if stack_frame_ids is None:
        return _error_response(request, 'Missing stack_frame_ids parameter')
    if not stack_frame_ids:
        return _error_response(request, 'Empty stack_frame_ids parameter')
    stack_frame_ids = stack_frame_ids.split(',')
    run_tag_filter = debug_data_provider.stack_frames_run_tag_filter(run, stack_frame_ids)
    blob_sequences = self._data_provider.read_blob_sequences(experiment_id=experiment, plugin_name=self.plugin_name, run_tag_filter=run_tag_filter)
    tag = next(iter(run_tag_filter.tags))
    try:
        return http_util.Respond(request, self._data_provider.read_blob(blob_key=blob_sequences[run][tag][0].blob_key), 'application/json')
    except errors.NotFoundError as e:
        return _error_response(request, str(e))