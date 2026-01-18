def process_network_proto(caffe_root, deploy_proto):
    """
    Runs the caffe upgrade tool on the prototxt to create a prototxt in the latest format.
    This enable us to work just with latest structures, instead of supporting all the variants

    :param caffe_root: link to caffe root folder, where the upgrade tool is located
    :param deploy_proto: name of the original prototxt file
    :return: name of new processed prototxt file
    """
    processed_deploy_proto = deploy_proto + '.processed'
    from shutil import copyfile
    copyfile(deploy_proto, processed_deploy_proto)
    import os
    upgrade_tool_command_line = caffe_root + '/build/tools/upgrade_net_proto_text.bin ' + processed_deploy_proto + ' ' + processed_deploy_proto
    os.system(upgrade_tool_command_line)
    return processed_deploy_proto