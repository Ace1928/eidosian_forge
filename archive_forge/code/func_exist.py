import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def exist(self, allctn=False):
    return self.docker_name() in self.get_containers(allctn=allctn)