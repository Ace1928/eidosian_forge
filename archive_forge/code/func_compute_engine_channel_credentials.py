import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
def compute_engine_channel_credentials(call_credentials):
    """Creates a compute engine channel credential.

    This credential can only be used in a GCP environment as it relies on
    a handshaker service. For more info about ALTS, see
    https://cloud.google.com/security/encryption-in-transit/application-layer-transport-security

    This channel credential is expected to be used as part of a composite
    credential in conjunction with a call credentials that authenticates the
    VM's default service account. If used with any other sort of call
    credential, the connection may suddenly and unexpectedly begin failing RPCs.
    """
    return ChannelCredentials(_cygrpc.channel_credentials_compute_engine(call_credentials._credentials))