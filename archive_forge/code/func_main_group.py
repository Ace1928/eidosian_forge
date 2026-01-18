import rasterio. But if you are using rasterio, you may profit from
import itertools
import logging
import sys
from click_plugins import with_plugins
import click
import cligj
from . import options
import rasterio
from rasterio.session import AWSSession
@with_plugins(itertools.chain(entry_points(group='rasterio.rio_commands'), entry_points(group='rasterio.rio_plugins')))
@click.group()
@cligj.verbose_opt
@cligj.quiet_opt
@click.option('--aws-profile', help='Select a profile from the AWS credentials file')
@click.option('--aws-no-sign-requests', is_flag=True, help='Make requests anonymously')
@click.option('--aws-requester-pays', is_flag=True, help='Requester pays data transfer costs')
@click.version_option(version=rasterio.__version__, message='%(version)s')
@click.option('--gdal-version', is_eager=True, is_flag=True, callback=gdal_version_cb)
@click.option('--show-versions', help='Show dependency versions', is_eager=True, is_flag=True, callback=show_versions_cb)
@click.pass_context
def main_group(ctx, verbose, quiet, aws_profile, aws_no_sign_requests, aws_requester_pays, gdal_version, show_versions):
    """Rasterio command line interface.
    """
    verbosity = verbose - quiet
    configure_logging(verbosity)
    ctx.obj = {}
    ctx.obj['verbosity'] = verbosity
    ctx.obj['aws_profile'] = aws_profile
    envopts = {'CPL_DEBUG': verbosity > 2}
    if aws_profile or aws_no_sign_requests or aws_requester_pays:
        ctx.obj['env'] = rasterio.Env(session=AWSSession(profile_name=aws_profile, aws_unsigned=aws_no_sign_requests, requester_pays=aws_requester_pays), **envopts)
    else:
        ctx.obj['env'] = rasterio.Env(**envopts)