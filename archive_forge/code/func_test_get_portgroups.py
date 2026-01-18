import collections
from unittest import mock
from oslo_vmware import dvs_util
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_portgroups(self):
    session = mock.Mock()
    dvs_moref = dvs_util.get_dvs_moref('dvs-123')
    pg_moref = vim_util.get_moref('dvportgroup-7', 'DistributedVirtualPortgroup')

    def session_invoke_api_side_effect(module, method, *args, **kwargs):
        if module == vim_util and method == 'get_object_properties':
            if ['portgroup'] in args:
                propSet = [DynamicProperty(name='portgroup', val=[[pg_moref]])]
                return [ObjectContent(obj=dvs_moref, propSet=propSet)]
            if ['name'] in args:
                propSet = [DynamicProperty(name='name', val='pg-name')]
                return [ObjectContent(obj=pg_moref, propSet=propSet)]
    session.invoke_api.side_effect = session_invoke_api_side_effect
    session._call_method.return_value = []
    pgs = dvs_util.get_portgroups(session, dvs_moref)
    result = [('pg-name', pg_moref)]
    self.assertEqual(result, pgs)