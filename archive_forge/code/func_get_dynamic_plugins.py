import logging
import pkg_resources
from tensorboard.plugins.audio import audio_plugin
from tensorboard.plugins.core import core_plugin
from tensorboard.plugins.custom_scalar import custom_scalars_plugin
from tensorboard.plugins.debugger_v2 import debugger_v2_plugin
from tensorboard.plugins.distribution import distributions_plugin
from tensorboard.plugins.graph import graphs_plugin
from tensorboard.plugins.histogram import histograms_plugin
from tensorboard.plugins.hparams import hparams_plugin
from tensorboard.plugins.image import images_plugin
from tensorboard.plugins.metrics import metrics_plugin
from tensorboard.plugins.pr_curve import pr_curves_plugin
from tensorboard.plugins.profile_redirect import profile_redirect_plugin
from tensorboard.plugins.scalar import scalars_plugin
from tensorboard.plugins.text import text_plugin
from tensorboard.plugins.mesh import mesh_plugin
from tensorboard.plugins.wit_redirect import wit_redirect_plugin
def get_dynamic_plugins():
    """Returns a list specifying TensorBoard's dynamically loaded plugins.

    A dynamic TensorBoard plugin is specified using entry_points [1] and it is
    the robust way to integrate plugins into TensorBoard.

    This list can be passed to the `tensorboard.program.TensorBoard` API.

    Returns:
      The list of dynamic plugins.

    :rtype: list[Type[base_plugin.TBLoader] | Type[base_plugin.TBPlugin]]

    [1]: https://packaging.python.org/specifications/entry-points/
    """
    return [entry_point.resolve() for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins')]