import os
import json
from shutil import copyfile, rmtree
from docker.tls import TLSConfig
from docker.errors import ContextException
from docker.context.config import get_meta_dir
from docker.context.config import get_meta_file
from docker.context.config import get_tls_dir
from docker.context.config import get_context_host
def is_docker_host(self):
    return self.context_type is None