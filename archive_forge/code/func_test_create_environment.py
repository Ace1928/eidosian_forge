import json
from tests.unit import AWSMockServiceTestCase
from boto.beanstalk.layer1 import Layer1
def test_create_environment(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_environment('application1', 'environment1', 'version1', '32bit Amazon Linux running Tomcat 7', option_settings=[('aws:autoscaling:launchconfiguration', 'Ec2KeyName', 'mykeypair'), ('aws:elasticbeanstalk:application:environment', 'ENVVAR', 'VALUE1')])
    self.assert_request_parameters({'Action': 'CreateEnvironment', 'ApplicationName': 'application1', 'EnvironmentName': 'environment1', 'TemplateName': '32bit Amazon Linux running Tomcat 7', 'ContentType': 'JSON', 'Version': '2010-12-01', 'VersionLabel': 'version1', 'OptionSettings.member.1.Namespace': 'aws:autoscaling:launchconfiguration', 'OptionSettings.member.1.OptionName': 'Ec2KeyName', 'OptionSettings.member.1.Value': 'mykeypair', 'OptionSettings.member.2.Namespace': 'aws:elasticbeanstalk:application:environment', 'OptionSettings.member.2.OptionName': 'ENVVAR', 'OptionSettings.member.2.Value': 'VALUE1'})