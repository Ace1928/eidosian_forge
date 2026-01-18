import os
def calc_ownership_spec(spec):
    """
    Calculates what a string spec means, returning (uid, username,
    gid, groupname), where there can be None values meaning no
    preference.

    The spec is a string like ``owner:group``.  It may use numbers
    instead of user/group names.  It may leave out ``:group``.  It may
    use '-' to mean any-user/any-group.

    """
    import grp
    import pwd
    user = group = None
    uid = gid = None
    if ':' in spec:
        user_spec, group_spec = spec.split(':', 1)
    else:
        user_spec, group_spec = (spec, '-')
    if user_spec == '-':
        user_spec = '0'
    if group_spec == '-':
        group_spec = '0'
    try:
        uid = int(user_spec)
    except ValueError:
        uid = pwd.getpwnam(user_spec)
        user = user_spec
    else:
        if not uid:
            uid = user = None
        else:
            user = pwd.getpwuid(uid).pw_name
    try:
        gid = int(group_spec)
    except ValueError:
        gid = grp.getgrnam(group_spec)
        group = group_spec
    else:
        if not gid:
            gid = group = None
        else:
            group = grp.getgrgid(gid).gr_name
    return (uid, user, gid, group)