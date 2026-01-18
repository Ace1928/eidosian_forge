import logging
import http.client as http_client
from ..constants import DEFAULT_SWARM_ADDR_POOL, DEFAULT_SWARM_SUBNET_SIZE
from .. import errors
from .. import types
from .. import utils
def create_swarm_spec(self, *args, **kwargs):
    """
        Create a :py:class:`docker.types.SwarmSpec` instance that can be used
        as the ``swarm_spec`` argument in
        :py:meth:`~docker.api.swarm.SwarmApiMixin.init_swarm`.

        Args:
            task_history_retention_limit (int): Maximum number of tasks
                history stored.
            snapshot_interval (int): Number of logs entries between snapshot.
            keep_old_snapshots (int): Number of snapshots to keep beyond the
                current snapshot.
            log_entries_for_slow_followers (int): Number of log entries to
                keep around to sync up slow followers after a snapshot is
                created.
            heartbeat_tick (int): Amount of ticks (in seconds) between each
                heartbeat.
            election_tick (int): Amount of ticks (in seconds) needed without a
                leader to trigger a new election.
            dispatcher_heartbeat_period (int):  The delay for an agent to send
                a heartbeat to the dispatcher.
            node_cert_expiry (int): Automatic expiry for nodes certificates.
            external_cas (:py:class:`list`): Configuration for forwarding
                signing requests to an external certificate authority. Use
                a list of :py:class:`docker.types.SwarmExternalCA`.
            name (string): Swarm's name
            labels (dict): User-defined key/value metadata.
            signing_ca_cert (str): The desired signing CA certificate for all
                swarm node TLS leaf certificates, in PEM format.
            signing_ca_key (str): The desired signing CA key for all swarm
                node TLS leaf certificates, in PEM format.
            ca_force_rotate (int): An integer whose purpose is to force swarm
                to generate a new signing CA certificate and key, if none have
                been specified.
            autolock_managers (boolean): If set, generate a key and use it to
                lock data stored on the managers.
            log_driver (DriverConfig): The default log driver to use for tasks
                created in the orchestrator.

        Returns:
            :py:class:`docker.types.SwarmSpec`

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> spec = client.api.create_swarm_spec(
              snapshot_interval=5000, log_entries_for_slow_followers=1200
            )
            >>> client.api.init_swarm(
              advertise_addr='eth0', listen_addr='0.0.0.0:5000',
              force_new_cluster=False, swarm_spec=spec
            )
        """
    ext_ca = kwargs.pop('external_ca', None)
    if ext_ca:
        kwargs['external_cas'] = [ext_ca]
    return types.SwarmSpec(self._version, *args, **kwargs)