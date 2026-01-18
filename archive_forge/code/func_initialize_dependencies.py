import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO
from unittest import *
import unittest as _unittest
import pytest as pytest
from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output
from unittest import mock
def initialize_dependencies(self):
    from pyomo.opt import check_available_solvers
    cls = self.__class__
    solvers_used = set(sum(list(cls.solver_dependencies.values()), []))
    available_solvers = check_available_solvers(*solvers_used)
    cls.solver_available = {solver_: solver_ in available_solvers for solver_ in solvers_used}
    cls.package_available = {}
    cls.package_modules = {}
    packages_used = set(sum(list(cls.package_dependencies.values()), []))
    for package_ in packages_used:
        pack, pack_avail = attempt_import(package_, defer_check=False)
        cls.package_available[package_] = pack_avail
        cls.package_modules[package_] = pack