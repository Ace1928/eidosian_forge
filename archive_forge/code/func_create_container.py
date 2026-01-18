from datetime import datetime
from .. import errors
from .. import utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..types import CancellableStream
from ..types import ContainerConfig
from ..types import EndpointConfig
from ..types import HostConfig
from ..types import NetworkingConfig
def create_container(self, image, command=None, hostname=None, user=None, detach=False, stdin_open=False, tty=False, ports=None, environment=None, volumes=None, network_disabled=False, name=None, entrypoint=None, working_dir=None, domainname=None, host_config=None, mac_address=None, labels=None, stop_signal=None, networking_config=None, healthcheck=None, stop_timeout=None, runtime=None, use_config_proxy=True, platform=None):
    """
        Creates a container. Parameters are similar to those for the ``docker
        run`` command except it doesn't support the attach options (``-a``).

        The arguments that are passed directly to this function are
        host-independent configuration options. Host-specific configuration
        is passed with the `host_config` argument. You'll normally want to
        use this method in combination with the :py:meth:`create_host_config`
        method to generate ``host_config``.

        **Port bindings**

        Port binding is done in two parts: first, provide a list of ports to
        open inside the container with the ``ports`` parameter, then declare
        bindings with the ``host_config`` parameter. For example:

        .. code-block:: python

            container_id = client.api.create_container(
                'busybox', 'ls', ports=[1111, 2222],
                host_config=client.api.create_host_config(port_bindings={
                    1111: 4567,
                    2222: None
                })
            )


        You can limit the host address on which the port will be exposed like
        such:

        .. code-block:: python

            client.api.create_host_config(
                port_bindings={1111: ('127.0.0.1', 4567)}
            )

        Or without host port assignment:

        .. code-block:: python

            client.api.create_host_config(port_bindings={1111: ('127.0.0.1',)})

        If you wish to use UDP instead of TCP (default), you need to declare
        ports as such in both the config and host config:

        .. code-block:: python

            container_id = client.api.create_container(
                'busybox', 'ls', ports=[(1111, 'udp'), 2222],
                host_config=client.api.create_host_config(port_bindings={
                    '1111/udp': 4567, 2222: None
                })
            )

        To bind multiple host ports to a single container port, use the
        following syntax:

        .. code-block:: python

            client.api.create_host_config(port_bindings={
                1111: [1234, 4567]
            })

        You can also bind multiple IPs to a single container port:

        .. code-block:: python

            client.api.create_host_config(port_bindings={
                1111: [
                    ('192.168.0.100', 1234),
                    ('192.168.0.101', 1234)
                ]
            })

        **Using volumes**

        Volume declaration is done in two parts. Provide a list of
        paths to use as mountpoints inside the container with the
        ``volumes`` parameter, and declare mappings from paths on the host
        in the ``host_config`` section.

        .. code-block:: python

            container_id = client.api.create_container(
                'busybox', 'ls', volumes=['/mnt/vol1', '/mnt/vol2'],
                host_config=client.api.create_host_config(binds={
                    '/home/user1/': {
                        'bind': '/mnt/vol2',
                        'mode': 'rw',
                    },
                    '/var/www': {
                        'bind': '/mnt/vol1',
                        'mode': 'ro',
                    }
                })
            )

        You can alternatively specify binds as a list. This code is equivalent
        to the example above:

        .. code-block:: python

            container_id = client.api.create_container(
                'busybox', 'ls', volumes=['/mnt/vol1', '/mnt/vol2'],
                host_config=client.api.create_host_config(binds=[
                    '/home/user1/:/mnt/vol2',
                    '/var/www:/mnt/vol1:ro',
                ])
            )

        **Networking**

        You can specify networks to connect the container to by using the
        ``networking_config`` parameter. At the time of creation, you can
        only connect a container to a single networking, but you
        can create more connections by using
        :py:meth:`~connect_container_to_network`.

        For example:

        .. code-block:: python

            networking_config = client.api.create_networking_config({
                'network1': client.api.create_endpoint_config(
                    ipv4_address='172.28.0.124',
                    aliases=['foo', 'bar'],
                    links=['container2']
                )
            })

            ctnr = client.api.create_container(
                img, command, networking_config=networking_config
            )

        Args:
            image (str): The image to run
            command (str or list): The command to be run in the container
            hostname (str): Optional hostname for the container
            user (str or int): Username or UID
            detach (bool): Detached mode: run container in the background and
                return container ID
            stdin_open (bool): Keep STDIN open even if not attached
            tty (bool): Allocate a pseudo-TTY
            ports (list of ints): A list of port numbers
            environment (dict or list): A dictionary or a list of strings in
                the following format ``["PASSWORD=xxx"]`` or
                ``{"PASSWORD": "xxx"}``.
            volumes (str or list): List of paths inside the container to use
                as volumes.
            network_disabled (bool): Disable networking
            name (str): A name for the container
            entrypoint (str or list): An entrypoint
            working_dir (str): Path to the working directory
            domainname (str): The domain name to use for the container
            host_config (dict): A dictionary created with
                :py:meth:`create_host_config`.
            mac_address (str): The Mac Address to assign the container
            labels (dict or list): A dictionary of name-value labels (e.g.
                ``{"label1": "value1", "label2": "value2"}``) or a list of
                names of labels to set with empty values (e.g.
                ``["label1", "label2"]``)
            stop_signal (str): The stop signal to use to stop the container
                (e.g. ``SIGINT``).
            stop_timeout (int): Timeout to stop the container, in seconds.
                Default: 10
            networking_config (dict): A networking configuration generated
                by :py:meth:`create_networking_config`.
            runtime (str): Runtime to use with this container.
            healthcheck (dict): Specify a test to perform to check that the
                container is healthy.
            use_config_proxy (bool): If ``True``, and if the docker client
                configuration file (``~/.docker/config.json`` by default)
                contains a proxy configuration, the corresponding environment
                variables will be set in the container being created.
            platform (str): Platform in the format ``os[/arch[/variant]]``.

        Returns:
            A dictionary with an image 'Id' key and a 'Warnings' key.

        Raises:
            :py:class:`docker.errors.ImageNotFound`
                If the specified image does not exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    if isinstance(volumes, str):
        volumes = [volumes]
    if isinstance(environment, dict):
        environment = utils.utils.format_environment(environment)
    if use_config_proxy:
        environment = self._proxy_configs.inject_proxy_environment(environment) or None
    config = self.create_container_config(image, command, hostname, user, detach, stdin_open, tty, ports, environment, volumes, network_disabled, entrypoint, working_dir, domainname, host_config, mac_address, labels, stop_signal, networking_config, healthcheck, stop_timeout, runtime)
    return self.create_container_from_config(config, name, platform)