import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography import x509
from cryptography.x509.oid import NameOID
from oslo_serialization import base64
from oslo_serialization import jsonutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
def config_cluster(cluster, cluster_template, cfg_dir, force=False, certs=None, use_keystone=False, direct_output=False):
    """Return and write configuration for the given cluster."""
    if cluster_template.coe == 'kubernetes':
        return _config_cluster_kubernetes(cluster, cluster_template, cfg_dir, force, certs, use_keystone, direct_output)
    elif cluster_template.coe == 'swarm' or cluster_template.coe == 'swarm-mode':
        return _config_cluster_swarm(cluster, cluster_template, cfg_dir, force, certs)