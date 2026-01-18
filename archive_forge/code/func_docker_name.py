import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def docker_name(self):
    if TEST_PREFIX == DEFAULT_TEST_PREFIX:
        return self.name
    return '{0}_{1}'.format(TEST_PREFIX, self.name)