from unittest import mock
import uuid
import stevedore
from keystone.api._shared import authentication
from keystone import auth
from keystone.auth.plugins import base
from keystone.auth.plugins import mapped
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import auth_plugins
@mock.patch.object(auth.core, '_get_auth_driver_manager')
def test_addition_auth_steps(self, stevedore_mock):
    simple_challenge_plugin = SimpleChallengeResponse()
    extension = stevedore.extension.Extension(name='simple_challenge', entry_point=None, plugin=None, obj=simple_challenge_plugin)
    test_manager = stevedore.DriverManager.make_test_instance(extension)
    stevedore_mock.return_value = test_manager
    self.useFixture(auth_plugins.ConfigAuthPlugins(self.config_fixture, methods=[METHOD_NAME]))
    self.useFixture(auth_plugins.LoadAuthPlugins(METHOD_NAME))
    auth_data = {'methods': [METHOD_NAME]}
    auth_data[METHOD_NAME] = {'test': 'test'}
    auth_data = {'identity': auth_data}
    auth_info = auth.core.AuthInfo.create(auth_data)
    auth_context = auth.core.AuthContext(method_names=[])
    try:
        with self.make_request():
            authentication.authenticate(auth_info, auth_context)
    except exception.AdditionalAuthRequired as e:
        self.assertIn('methods', e.authentication)
        self.assertIn(METHOD_NAME, e.authentication['methods'])
        self.assertIn(METHOD_NAME, e.authentication)
        self.assertIn('challenge', e.authentication[METHOD_NAME])
    auth_data = {'methods': [METHOD_NAME]}
    auth_data[METHOD_NAME] = {'response': EXPECTED_RESPONSE}
    auth_data = {'identity': auth_data}
    auth_info = auth.core.AuthInfo.create(auth_data)
    auth_context = auth.core.AuthContext(method_names=[])
    with self.make_request():
        authentication.authenticate(auth_info, auth_context)
    self.assertEqual(DEMO_USER_ID, auth_context['user_id'])
    auth_data = {'methods': [METHOD_NAME]}
    auth_data[METHOD_NAME] = {'response': uuid.uuid4().hex}
    auth_data = {'identity': auth_data}
    auth_info = auth.core.AuthInfo.create(auth_data)
    auth_context = auth.core.AuthContext(method_names=[])
    with self.make_request():
        self.assertRaises(exception.Unauthorized, authentication.authenticate, auth_info, auth_context)