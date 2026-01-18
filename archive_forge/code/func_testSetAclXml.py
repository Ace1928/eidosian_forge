import binascii
import re
from six import StringIO
from boto import storage_uri
from boto.exception import BotoClientError
from boto.gs.acl import SupportedPermissions as perms
from tests.integration.gs.testcase import GSTestCase
def testSetAclXml(self):
    """Ensures that calls to the set_xml_acl functions succeed."""
    b = self._MakeBucket()
    k = b.new_key('obj')
    k.set_contents_from_string('stringdata')
    bucket_uri = storage_uri('gs://%s/' % b.name)
    bucket_uri.object_name = 'obj'
    bucket_acl = bucket_uri.get_acl()
    bucket_uri.object_name = None
    all_users_read_permission = "<Entry><Scope type='AllUsers'/><Permission>READ</Permission></Entry>"
    acl_string = re.sub('</Entries>', all_users_read_permission + '</Entries>', bucket_acl.to_xml())
    acl_no_owner_string = re.sub('<Owner>.*</Owner>', '', acl_string)
    bucket_uri.set_xml_acl(acl_string, 'obj')
    bucket_uri.set_xml_acl(acl_no_owner_string)
    bucket_uri.set_def_xml_acl(acl_no_owner_string)
    new_obj_acl_string = k.get_acl().to_xml()
    new_bucket_acl_string = bucket_uri.get_acl().to_xml()
    new_bucket_def_acl_string = bucket_uri.get_def_acl().to_xml()
    self.assertRegexpMatches(new_obj_acl_string, 'AllUsers')
    self.assertRegexpMatches(new_bucket_acl_string, 'AllUsers')
    self.assertRegexpMatches(new_bucket_def_acl_string, 'AllUsers')