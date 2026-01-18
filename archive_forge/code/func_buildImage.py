from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import os
def buildImage(opt):
    dpath = os.path.join(opt['datapath'], 'COCO-IMG-2014')
    version = '1'
    if not build_data.built(dpath, version_string=version):
        print('[building image data: ' + dpath + ']')
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        for downloadable_file in RESOURCES[:3]:
            downloadable_file.download_file(dpath)
        build_data.mark_done(dpath, version_string=version)