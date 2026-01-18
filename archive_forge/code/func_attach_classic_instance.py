from boto.ec2.ec2object import TaggedEC2Object
def attach_classic_instance(self, instance_id, groups, dry_run=False):
    """
        Links  an EC2-Classic instance to a ClassicLink-enabled VPC through one
        or more of the VPC's security groups. You cannot link an EC2-Classic
        instance to more than one VPC at a time. You can only link an instance
        that's in the running state. An instance is automatically unlinked from
        a VPC when it's stopped. You can link it to the VPC again when you
        restart it.

        After you've linked an instance, you cannot  change  the VPC security
        groups  that are associated with it. To change the security groups, you
        must first unlink the instance, and then link it again.

        Linking your instance to a VPC is sometimes referred  to  as  attaching
        your instance.

        :type intance_id: str
        :param instance_is: The ID of a ClassicLink-enabled VPC.

        :tye groups: list
        :param groups: The ID of one or more of the VPC's security groups.
            You cannot specify security groups from a different VPC. The
            members of the list can be
            :class:`boto.ec2.securitygroup.SecurityGroup` objects or
            strings of the id's of the security groups.

        :type dry_run: bool
        :param dry_run: Set to True if the operation should not actually run.

        :rtype: bool
        :return: True if successful
        """
    return self.connection.attach_classic_link_vpc(vpc_id=self.id, instance_id=instance_id, groups=groups, dry_run=dry_run)