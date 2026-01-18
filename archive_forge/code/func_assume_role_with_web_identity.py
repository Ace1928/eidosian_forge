from boto.connection import AWSQueryConnection
from boto.provider import Provider, NO_CREDENTIALS_PROVIDED
from boto.regioninfo import RegionInfo
from boto.sts.credentials import Credentials, FederationToken, AssumedRole
from boto.sts.credentials import DecodeAuthorizationMessage
import boto
import boto.utils
import datetime
import threading
def assume_role_with_web_identity(self, role_arn, role_session_name, web_identity_token, provider_id=None, policy=None, duration_seconds=None):
    """
        Returns a set of temporary security credentials for users who
        have been authenticated in a mobile or web application with a
        web identity provider, such as Login with Amazon, Facebook, or
        Google. `AssumeRoleWithWebIdentity` is an API call that does
        not require the use of AWS security credentials. Therefore,
        you can distribute an application (for example, on mobile
        devices) that requests temporary security credentials without
        including long-term AWS credentials in the application or by
        deploying server-based proxy services that use long-term AWS
        credentials. For more information, see `Creating a Mobile
        Application with Third-Party Sign-In`_ in AWS Security Token
        Service .

        The temporary security credentials consist of an access key
        ID, a secret access key, and a security token. Applications
        can use these temporary security credentials to sign calls to
        AWS service APIs. The credentials are valid for the duration
        that you specified when calling `AssumeRoleWithWebIdentity`,
        which can be from 900 seconds (15 minutes) to 3600 seconds (1
        hour). By default, the temporary security credentials are
        valid for 1 hour.

        The temporary security credentials that are returned from the
        `AssumeRoleWithWebIdentity` response have the permissions that
        are associated with the access policy of the role being
        assumed. You can further restrict the permissions of the
        temporary security credentials by passing a policy in the
        request. The resulting permissions are an intersection of the
        role's access policy and the policy that you passed. These
        policies and any applicable resource-based policies are
        evaluated when calls to AWS service APIs are made using the
        temporary security credentials.

        Before your application can call `AssumeRoleWithWebIdentity`,
        you must have an identity token from a supported identity
        provider and create a role that the application can assume.
        The role that your application assumes must trust the identity
        provider that is associated with the identity token. In other
        words, the identity provider must be specified in the role's
        trust policy. For more information, see ` Creating Temporary
        Security Credentials for Mobile Apps Using Third-Party
        Identity Providers`_.

        :type role_arn: string
        :param role_arn: The Amazon Resource Name (ARN) of the role that the
            caller is assuming.

        :type role_session_name: string
        :param role_session_name: An identifier for the assumed role session.
            Typically, you pass the name or identifier that is associated with
            the user who is using your application. That way, the temporary
            security credentials that your application will use are associated
            with that user. This session name is included as part of the ARN
            and assumed role ID in the `AssumedRoleUser` response element.

        :type web_identity_token: string
        :param web_identity_token: The OAuth 2.0 access token or OpenID Connect
            ID token that is provided by the identity provider. Your
            application must get this token by authenticating the user who is
            using your application with a web identity provider before the
            application makes an `AssumeRoleWithWebIdentity` call.

        :type provider_id: string
        :param provider_id: Specify this value only for OAuth access tokens. Do
            not specify this value for OpenID Connect ID tokens, such as
            `accounts.google.com`. This is the fully-qualified host component
            of the domain name of the identity provider. Do not include URL
            schemes and port numbers. Currently, `www.amazon.com` and
            `graph.facebook.com` are supported.

        :type policy: string
        :param policy: A supplemental policy that is associated with the
            temporary security credentials from the `AssumeRoleWithWebIdentity`
            call. The resulting permissions of the temporary security
            credentials are an intersection of this policy and the access
            policy that is associated with the role. Use this policy to further
            restrict the permissions of the temporary security credentials.

        :type duration_seconds: integer
        :param duration_seconds: The duration, in seconds, of the role session.
            The value can range from 900 seconds (15 minutes) to 3600 seconds
            (1 hour). By default, the value is set to 3600 seconds.

        """
    params = {'RoleArn': role_arn, 'RoleSessionName': role_session_name, 'WebIdentityToken': web_identity_token}
    if provider_id is not None:
        params['ProviderId'] = provider_id
    if policy is not None:
        params['Policy'] = policy
    if duration_seconds is not None:
        params['DurationSeconds'] = duration_seconds
    return self.get_object('AssumeRoleWithWebIdentity', params, AssumedRole, verb='POST')