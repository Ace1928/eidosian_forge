import functools
from heat.tests.convergence.framework import reality
from heat.tests.convergence.framework import scenario_template
from oslo_log import log as logging
def scenario_globals(procs, testcase):
    return {'test': testcase, 'reality': reality.reality, 'verify': functools.partial(verify, testcase, reality.reality), 'Template': scenario_template.Template, 'RsrcDef': scenario_template.RsrcDef, 'GetRes': scenario_template.GetRes, 'GetAtt': scenario_template.GetAtt, 'engine': procs.engine, 'worker': procs.worker}