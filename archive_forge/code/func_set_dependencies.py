import hashlib
import os
import random
from operator import attrgetter
import gyp.common
def set_dependencies(self, dependencies):
    self.dependencies = list(dependencies or [])