import logging
import threading
from io import BytesIO
import awscrt.http
import awscrt.s3
import botocore.awsrequest
import botocore.session
from awscrt.auth import (
from awscrt.io import (
from awscrt.s3 import S3Client, S3RequestTlsMode, S3RequestType
from botocore import UNSIGNED
from botocore.compat import urlsplit
from botocore.config import Config
from botocore.exceptions import NoCredentialsError
from s3transfer.constants import MB
from s3transfer.exceptions import TransferNotDoneError
from s3transfer.futures import BaseTransferFuture, BaseTransferMeta
from s3transfer.utils import (
def create_s3_crt_client(region, crt_credentials_provider=None, num_threads=None, target_throughput=None, part_size=8 * MB, use_ssl=True, verify=None):
    """
    :type region: str
    :param region: The region used for signing

    :type crt_credentials_provider:
        Optional[awscrt.auth.AwsCredentialsProvider]
    :param crt_credentials_provider: CRT AWS credentials provider
        to use to sign requests. If not set, requests will not be signed.

    :type num_threads: Optional[int]
    :param num_threads: Number of worker threads generated. Default
        is the number of processors in the machine.

    :type target_throughput: Optional[int]
    :param target_throughput: Throughput target in bytes per second.
        By default, CRT will automatically attempt to choose a target
        throughput that matches the system's maximum network throughput.
        Currently, if CRT is unable to determine the maximum network
        throughput, a fallback target throughput of ``1_250_000_000`` bytes
        per second (which translates to 10 gigabits per second, or 1.16
        gibibytes per second) is used. To set a specific target
        throughput, set a value for this parameter.

    :type part_size: Optional[int]
    :param part_size: Size, in Bytes, of parts that files will be downloaded
        or uploaded in.

    :type use_ssl: boolean
    :param use_ssl: Whether or not to use SSL.  By default, SSL is used.
        Note that not all services support non-ssl connections.

    :type verify: Optional[boolean/string]
    :param verify: Whether or not to verify SSL certificates.
        By default SSL certificates are verified.  You can provide the
        following values:

        * False - do not validate SSL certificates.  SSL will still be
            used (unless use_ssl is False), but SSL certificates
            will not be verified.
        * path/to/cert/bundle.pem - A filename of the CA cert bundle to
            use. Specify this argument if you want to use a custom CA cert
            bundle instead of the default one on your system.
    """
    event_loop_group = EventLoopGroup(num_threads)
    host_resolver = DefaultHostResolver(event_loop_group)
    bootstrap = ClientBootstrap(event_loop_group, host_resolver)
    tls_connection_options = None
    tls_mode = S3RequestTlsMode.ENABLED if use_ssl else S3RequestTlsMode.DISABLED
    if verify is not None:
        tls_ctx_options = TlsContextOptions()
        if verify:
            tls_ctx_options.override_default_trust_store_from_path(ca_filepath=verify)
        else:
            tls_ctx_options.verify_peer = False
        client_tls_option = ClientTlsContext(tls_ctx_options)
        tls_connection_options = client_tls_option.new_connection_options()
    target_gbps = _get_crt_throughput_target_gbps(provided_throughput_target_bytes=target_throughput)
    return S3Client(bootstrap=bootstrap, region=region, credential_provider=crt_credentials_provider, part_size=part_size, tls_mode=tls_mode, tls_connection_options=tls_connection_options, throughput_target_gbps=target_gbps, enable_s3express=True)