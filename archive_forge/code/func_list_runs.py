import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def list_runs(self, ctx=None, *, experiment_id):
    """List runs available.

        Args:
          experiment_id: currently unused, because the backing
            DebuggerV2EventMultiplexer does not accommodate multiple experiments.

        Returns:
          Run names as a list of str.
        """
    return [provider.Run(run_id=run, run_name=run, start_time=self._get_first_event_timestamp(run)) for run in self._multiplexer.Runs()]