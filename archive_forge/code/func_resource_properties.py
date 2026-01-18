from unittest import mock
from heat.common import exception
from heat.db import api as db_api
from heat.tests import utils
def resource_properties(self, res, prop_name):
    res_data = db_api.resource_data_get_by_key(self.cntxt, res.id, prop_name)
    return res_data.value