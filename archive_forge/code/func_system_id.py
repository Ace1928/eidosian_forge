from os_ken.controller import event as os_ken_event
from os_ken.controller import handler
@property
def system_id(self):
    return self.client.system_id