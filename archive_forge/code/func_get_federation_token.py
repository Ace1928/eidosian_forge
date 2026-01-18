from boto.connection import AWSQueryConnection
from boto.provider import Provider, NO_CREDENTIALS_PROVIDED
from boto.regioninfo import RegionInfo
from boto.sts.credentials import Credentials, FederationToken, AssumedRole
from boto.sts.credentials import DecodeAuthorizationMessage
import boto
import boto.utils
import datetime
import threading
def get_federation_token(self, name, duration=None, policy=None):
    """
        Returns a set of temporary security credentials (consisting of
        an access key ID, a secret access key, and a security token)
        for a federated user. A typical use is in a proxy application
        that is getting temporary security credentials on behalf of
        distributed applications inside a corporate network. Because
        you must call the `GetFederationToken` action using the long-
        term security credentials of an IAM user, this call is
        appropriate in contexts where those credentials can be safely
        stored, usually in a server-based application.

        **Note:** Do not use this call in mobile applications or
        client-based web applications that directly get temporary
        security credentials. For those types of applications, use
        `AssumeRoleWithWebIdentity`.

        The `GetFederationToken` action must be called by using the
        long-term AWS security credentials of the AWS account or an
        IAM user. Credentials that are created by IAM users are valid
        for the specified duration, between 900 seconds (15 minutes)
        and 129600 seconds (36 hours); credentials that are created by
        using account credentials have a maximum duration of 3600
        seconds (1 hour).

        The permissions that are granted to the federated user are the
        intersection of the policy that is passed with the
        `GetFederationToken` request and policies that are associated
        with of the entity making the `GetFederationToken` call.

        For more information about how permissions work, see
        `Controlling Permissions in Temporary Credentials`_ in Using
        Temporary Security Credentials . For information about using
        `GetFederationToken` to create temporary security credentials,
        see `Creating Temporary Credentials to Enable Access for
        Federated Users`_ in Using Temporary Security Credentials .

        :type name: string
        :param name: The name of the federated user. The name is used as an
            identifier for the temporary security credentials (such as `Bob`).
            For example, you can reference the federated user name in a
            resource-based policy, such as in an Amazon S3 bucket policy.

        :type policy: string
        :param policy: A policy that specifies the permissions that are granted
            to the federated user. By default, federated users have no
            permissions; they do not inherit any from the IAM user. When you
            specify a policy, the federated user's permissions are intersection
            of the specified policy and the IAM user's policy. If you don't
            specify a policy, federated users can only access AWS resources
            that explicitly allow those federated users in a resource policy,
            such as in an Amazon S3 bucket policy.

        :type duration: integer
        :param duration: The duration, in seconds, that the session
            should last. Acceptable durations for federation sessions range
            from 900 seconds (15 minutes) to 129600 seconds (36 hours), with
            43200 seconds (12 hours) as the default. Sessions for AWS account
            owners are restricted to a maximum of 3600 seconds (one hour). If
            the duration is longer than one hour, the session for AWS account
            owners defaults to one hour.

        """
    params = {'Name': name}
    if duration:
        params['DurationSeconds'] = duration
    if policy:
        params['Policy'] = policy
    return self.get_object('GetFederationToken', params, FederationToken, verb='POST')