import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
def format_tiers(self, line):
    if 'meV' in line:
        assert line[0] == '#'
        if 'tier' in line and 'Further' not in line:
            tier = line.split(' tier')[0]
            tier = tier.split('"')[-1]
            current_tier = self.translate_tier(tier)
            if current_tier == self.targettier:
                self.foundtarget = True
            elif current_tier > self.targettier:
                self.do_uncomment = False
        else:
            self.do_uncomment = False
        return line
    elif self.do_uncomment and line[0] == '#':
        return line[1:]
    elif not self.do_uncomment and line[0] != '#':
        return '#' + line
    else:
        return line