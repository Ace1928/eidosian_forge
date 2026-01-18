import uuid
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def frozen_definition(self):
    return self.t.freeze(properties=properties.Properties(schema={}, data={}))