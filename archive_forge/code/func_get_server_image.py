from heat_integrationtests.functional import functional_base
def get_server_image(server_id):
    server = self.compute_client.servers.get(server_id)
    return server.image['id']