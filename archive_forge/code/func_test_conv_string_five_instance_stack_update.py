from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_conv_string_five_instance_stack_update(self, mock_cr):
    stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
    stack.store()
    stack.converge_stack(template=stack.t, action=stack.CREATE)
    curr_stack_db = stack_object.Stack.get_by_id(stack.context, stack.id)
    curr_stack = parser.Stack.load(curr_stack_db._context, stack=curr_stack_db)
    t2 = template_format.parse(tools.string_template_five_update)
    template2 = templatem.Template(t2, env=environment.Environment({'KeyName2': 'test2'}))
    self.stack = stack
    with mock.patch.object(parser.Stack, 'db_active_resources_get', side_effect=self._mock_convg_db_update_requires):
        curr_stack.thread_group_mgr = tools.DummyThreadGroupManager()
        curr_stack.converge_stack(template=template2, action=stack.UPDATE)
    self.assertIsNotNone(curr_stack.ext_rsrcs_db)
    deps = curr_stack.convergence_dependencies
    self.assertEqual([((3, False), (1, False)), ((3, False), (2, False)), ((4, False), (3, False)), ((4, False), (4, True)), ((5, False), (3, False)), ((5, False), (5, True)), ((6, True), (8, True)), ((7, True), (8, True)), ((8, True), (4, True)), ((8, True), (5, True))], sorted(deps._graph.edges()))
    stack_db = stack_object.Stack.get_by_id(curr_stack.context, curr_stack.id)
    self.assertIsNotNone(stack_db.raw_template_id)
    self.assertIsNotNone(stack_db.current_traversal)
    self.assertIsNotNone(stack_db.prev_raw_template_id)
    self.assertTrue(stack_db.convergence)
    self.assertEqual(sorted([[[7, True], [8, True]], [[8, True], [5, True]], [[8, True], [4, True]], [[6, True], [8, True]], [[3, False], [2, False]], [[3, False], [1, False]], [[5, False], [3, False]], [[5, False], [5, True]], [[4, False], [3, False]], [[4, False], [4, True]]]), sorted(stack_db.current_deps['edges']))
    '\n        To visualize:\n\n        G(7, True)       H(6, True)\n            \\                 /\n              \\             /           B(4, False)   A(5, False)\n                \\         /               /       \\  /    /\n                  \\     /            /           /\n               F(8, True)       /             /     \\  /\n                    /  \\    /             /     C(3, False)\n                  /      \\            /            /    \\\n                /     /    \\      /\n              /    /         \\ /                /          \\\n        B(4, True)      A(5, True)       D(2, False)    E(1, False)\n\n        Leaves are at the bottom\n        '
    for entity_id in [8, 7, 6, 5, 4, stack_db.id]:
        sync_point = sync_point_object.SyncPoint.get_by_key(stack_db._context, entity_id, stack_db.current_traversal, True)
        self.assertIsNotNone(sync_point)
        self.assertEqual(stack_db.id, sync_point.stack_id)
    for entity_id in [5, 4, 3, 2, 1]:
        sync_point = sync_point_object.SyncPoint.get_by_key(stack_db._context, entity_id, stack_db.current_traversal, False)
        self.assertIsNotNone(sync_point)
        self.assertEqual(stack_db.id, sync_point.stack_id)
    leaves = set(stack.convergence_dependencies.leaves())
    expected_calls = []
    for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
        expected_calls.append(mock.call.worker_client.WorkerClient.check_resource(stack.context, rsrc_id, stack.current_traversal, {'input_data': {}}, is_update, None, False))
    leaves = set(curr_stack.convergence_dependencies.leaves())
    for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
        expected_calls.append(mock.call.worker_client.WorkerClient.check_resource(curr_stack.context, rsrc_id, curr_stack.current_traversal, {'input_data': {}}, is_update, None, False))
    self.assertEqual(expected_calls, mock_cr.mock_calls)