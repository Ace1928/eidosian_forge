import boto
from boto.cloudformation.stack import Stack, StackSummary, StackEvent
from boto.cloudformation.stack import StackResource, StackResourceSummary
from boto.cloudformation.template import Template
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
def estimate_template_cost(self, template_body=None, template_url=None, parameters=None):
    """
        Returns the estimated monthly cost of a template. The return
        value is an AWS Simple Monthly Calculator URL with a query
        string that describes the resources required to run the
        template.

        :type template_body: string
        :param template_body: Structure containing the template body. (For more
            information, go to `Template Anatomy`_ in the AWS CloudFormation
            User Guide.)
        Conditional: You must pass `TemplateBody` or `TemplateURL`. If both are
            passed, only `TemplateBody` is used.

        :type template_url: string
        :param template_url: Location of file containing the template body. The
            URL must point to a template located in an S3 bucket in the same
            region as the stack. For more information, go to `Template
            Anatomy`_ in the AWS CloudFormation User Guide.
        Conditional: You must pass `TemplateURL` or `TemplateBody`. If both are
            passed, only `TemplateBody` is used.

        :type parameters: list
        :param parameters: A list of key/value tuples that specify input
            parameters for the template.

        :rtype: string
        :returns: URL to pre-filled cost calculator
        """
    params = {'ContentType': 'JSON'}
    if template_body is not None:
        params['TemplateBody'] = template_body
    if template_url is not None:
        params['TemplateURL'] = template_url
    if parameters and len(parameters) > 0:
        for i, (key, value) in enumerate(parameters):
            params['Parameters.member.%d.ParameterKey' % (i + 1)] = key
            params['Parameters.member.%d.ParameterValue' % (i + 1)] = value
    response = self._do_request('EstimateTemplateCost', params, '/', 'POST')
    return response['EstimateTemplateCostResponse']['EstimateTemplateCostResult']['Url']