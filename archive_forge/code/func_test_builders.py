from tests.unit import unittest
import xml.dom.minidom
import xml.sax
from boto.s3.website import WebsiteConfiguration
from boto.s3.website import RedirectLocation
from boto.s3.website import RoutingRules
from boto.s3.website import Condition
from boto.s3.website import RoutingRules
from boto.s3.website import RoutingRule
from boto.s3.website import Redirect
from boto import handler
def test_builders(self):
    x = pretty_print_xml
    rules = RoutingRules()
    condition = Condition(http_error_code=404)
    redirect = Redirect(hostname='example.com', replace_key_prefix='report-404/')
    rules.add_rule(RoutingRule(condition, redirect))
    xml = rules.to_xml()
    rules2 = RoutingRules().add_rule(RoutingRule.when(http_error_code=404).then_redirect(hostname='example.com', replace_key_prefix='report-404/'))
    xml2 = rules2.to_xml()
    self.assertEqual(x(xml), x(xml2))