from keystone.server.flask import core as flask_core
def initialize_public_application():
    return flask_core.initialize_application(name='public', config_files=flask_core._get_config_files())