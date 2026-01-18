import os
from pyomo.common import unittest, Executable
from pyomo.common.errors import DeveloperError
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.plugins.nl_writer import NLWriter
from pyomo.contrib.solver import ipopt
def test_create_command_line(self):
    opt = ipopt.Ipopt()
    result = opt._create_command_line('myfile', opt.config, False)
    self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL'])
    opt = ipopt.Ipopt(solver_options={'max_iter': 4})
    result = opt._create_command_line('myfile', opt.config, False)
    self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'max_iter=4'])
    opt = ipopt.Ipopt(solver_options={'max_iter': 4}, time_limit=10)
    result = opt._create_command_line('myfile', opt.config, False)
    self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'max_iter=4', 'max_cpu_time=10.0'])
    opt = ipopt.Ipopt(solver_options={'max_iter': 4, 'max_cpu_time': 10})
    result = opt._create_command_line('myfile', opt.config, False)
    self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'max_cpu_time=10', 'max_iter=4'])
    result = opt._create_command_line('myfile', opt.config, True)
    self.assertEqual(result, [str(opt.config.executable), 'myfile.nl', '-AMPL', 'option_file_name=myfile.opt', 'max_cpu_time=10', 'max_iter=4'])
    opt = ipopt.Ipopt(solver_options={'max_iter': 4, 'option_file_name': 'myfile.opt'})
    with self.assertRaises(ValueError):
        result = opt._create_command_line('myfile', opt.config, False)