from openstack.accelerator.v2._proxy import Proxy
def bind_accelerator_request(self, uuid, properties):
    """Bind an accelerator to VM.

        :param uuid: The uuid of the accelerator_request to be binded.
        :param properties: The info of VM that will bind the accelerator.
        :returns: True if bind succeeded, False otherwise.
        """
    accelerator_request = self.accelerator.get_accelerator_request(uuid)
    if accelerator_request is None:
        self.log.debug('accelerator_request %s not found for unbinding', uuid)
        return False
    return self.accelerator.update_accelerator_request(uuid, properties)