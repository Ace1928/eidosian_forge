import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.data import provider
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.pr_curve import metadata
@wrappers.Request.application
def tags_route(self, request):
    """A route (HTTP handler) that returns a response with tags.

        Returns:
          A response that contains a JSON object. The keys of the object
          are all the runs. Each run is mapped to a (potentially empty) dictionary
          whose keys are tags associated with run and whose values are metadata
          (dictionaries).

          The metadata dictionaries contain 2 keys:
            - displayName: For the display name used atop visualizations in
                TensorBoard.
            - description: The description that appears near visualizations upon the
                user hovering over a certain icon.
        """
    ctx = plugin_util.context(request.environ)
    experiment = plugin_util.experiment_id(request.environ)
    return http_util.Respond(request, self.tags_impl(ctx, experiment), 'application/json')